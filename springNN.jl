using DifferentialEquations
using LinearAlgebra
using Makie, GLMakie
using Colors

# ── 1. Build the network ─────────────────────────────────────────────────────
"""
    build_network(layer_sizes::Vector{Int})

Given a list of layer sizes [n₁,n₂,…,n_L], returns:
- N = total nodes
- layers = vector of 1:N giving layer index per node
- neighbors = Vector{Vector{Int}} adjacency list
"""
function build_network(layer_sizes)
    # assign global indices
    idx = 1
    layers = Int[]
    for (ℓ,n) in enumerate(layer_sizes)
        append!(layers, fill(ℓ,n))
    end
    N = sum(layer_sizes)
    # build neighbors: complete bipartite between ℓ and ℓ+1
    neighbors = [Int[] for i in 1:N]
    offset = cumsum(vcat(0,layer_sizes))
    for ℓ in 1:length(layer_sizes)-1
        for i in 1:layer_sizes[ℓ], j in 1:layer_sizes[ℓ+1]
            u = offset[ℓ] + i
            v = offset[ℓ+1] + j
            push!(neighbors[u],v)
            push!(neighbors[v],u)
        end
    end
    return N, layers, neighbors
end

# ── 2. ODE definition ───────────────────────────────────────────────────────
"""
    spring_mass_ode!(du,u,p,t)

u = [ x₁…x_N, v₁…v_N ]
p = (; m, k_sink, k_cpl, c, neighbors )
"""
function spring_mass_ode!(du,u,p,t)
    N = length(p.m)
    x = @view u[1:N]
    v = @view u[N+1:2N]
    dx = @view du[1:N]
    dv = @view du[N+1:2N]
    # x' = v
    dx .= v
    # build forces
    @inbounds for i in 1:N
        # sink + damping
        f = -p.k_sink[i]*x[i] - p.c[i]*v[i]
        # springs to neighbors
        for j in p.neighbors[i]
            f += -p.k_cpl * (x[i]-x[j])
        end
        dv[i] = f / p.m[i]
    end
end

# ── 3. Energy calculations ──────────────────────────────────────────────────
"""
    compute_energies(sol, p)

Given a solution object `sol` from ODE solve, returns:
- E_local::Matrix  size (N, length(t))
- E_total::Vector length(t)
"""
function compute_energies(sol, p)
    N     = length(p.m)
    # pull out the full solution as a matrix
    M     = Array(sol)         # size (2N) × (ntime)
    xs    = M[1:N,   :]        # positions
    vs    = M[N+1:2N, :]       # velocities
    ntime = length(sol.t)

    # kinetic & sink
    Ekin  = 0.5 .* p.m    .* (vs .^ 2)
    Esink = 0.5 .* p.k_sink .* (xs .^ 2)

    # coupling energy per node (half spring energy per endpoint)
    Ecpl  = zeros(N, ntime)
    @inbounds for i in 1:N, ti in 1:ntime
        for j in p.neighbors[i]
            Ecpl[i,ti] += 0.5 * p.k_cpl * (xs[i,ti] - xs[j,ti])^2
        end
    end

    Elocal = Ekin .+ Esink .+ Ecpl
    Etotal = sum(Elocal, dims=1)[:]

    return Elocal, Etotal
end

# ── 4. Static plots ─────────────────────────────────────────────────────────
function plot_all_nodes(ts, Elocal, Etotal)
    fig = Figure(size=(800,400))
    ax = Axis(fig[1,1], xlabel="t", ylabel="Energy", title="All Node Energies")
    lines!(ax, ts, Etotal, color=:black, label="Total", linewidth=2)
    for i in 1:size(Elocal,1)
        lines!(ax, ts, Elocal[i,:], linestyle=:dash, label="Node $i")
    end
    axislegend(ax)
    fig
end

function plot_io_nodes(ts, Elocal, layer_assign; layers=(1,length(layer_assign)))
    fig = Figure(size=(800,300))
    ax1 = Axis(fig[1,1], xlabel="t", ylabel="Energy", title="Input Layer Energies")
    ax2 = Axis(fig[2,1], xlabel="t", ylabel="Energy", title="Output Layer Energies")
    inputs  = findall(x->x==layers[1], layer_assign)
    outputs = findall(x->x==layers[2], layer_assign)
    for i in inputs
        lines!(ax1, ts, Elocal[i,:], label="Node $i")
    end
    for j in outputs
        lines!(ax2, ts, Elocal[j,:], label="Node $j")
    end
    axislegend(ax1); axislegend(ax2)
    fig
end

function static_network_plot(layer_sizes::Vector{Int})
    fig = Figure(resolution = (600, 300))
    ax = Axis(fig[1,1];
        aspect         = DataAspect(),    # equal scaling x vs y
        xticks         = nothing,         # no tick marks
        yticks         = nothing,         # no tick marks
        xgridvisible   = false,
        ygridvisible   = false,
        showaxis       = false            # no frame/axis decorations
    )

    # build positions exactly as in the animation:
    N, layers, neighbors = build_network(layer_sizes)
    positions = Vector{Point2f}(undef, N)
    idx = 1
    for (ℓ, n) in enumerate(layer_sizes)
        ys = range(0, -1, length=n)
        for i in 1:n
            positions[idx] = Point2f(ℓ, ys[i])
            idx += 1
        end
    end

    # draw springs between layer ℓ and ℓ+1 only once:
    for i in 1:N, j in neighbors[i]
        if layers[j] == layers[i] + 1
            lines!(ax,
                [positions[i].x, positions[j].x],
                [positions[i].y, positions[j].y],
                color = :gray
            )
        end
    end

    # draw the masses / nodes
    scatter!(ax,
        getindex.(positions, 1),
        getindex.(positions, 2),
        color = :blue,
        markersize = 12
    )

    # label each node by its global index
    for i in 1:N
        text!(ax, string(i),
            position = (positions[i].x, positions[i].y),
            align = (:center, :center),
            color = :white,
            fontsize = 14
        )
    end

    return fig
end

# ── 5. GLMakie animation ────────────────────────────────────────────────────
function animate_network(sol, p, layer_sizes)
    scenes = Scene(size=(600,400), camera=campixel!)
    N = length(p.m)
    # initial positions in x-y plane
    offsets = cumsum(vcat(0,layer_sizes))
    positions = [ Point3f0(l, 0.0, 0.0) for l in repeat(1:length(layer_sizes), layer_sizes) ]
    # spheres
    sc = [ mesh!(scenes, Sphere(Point3f0(positions[i]), 0.1), color=:blue) for i in 1:N ]
    ts = sol.t
    Elocal, _ = compute_energies(sol,p)
    # animate
    @async for (k,t) in enumerate(ts)
        for i in 1:N
            # map energy to z-offset
            z = clamp(Elocal[i,k], 0, maximum(Elocal)) 
            sc[i].transformation.translation = Point3f0(positions[i][1], positions[i][2], 0.2*z)
        end
        sleep(1/60)  # ~60 fps
    end
    scenes
end

# ── Example usage ──────────────────────────────────────────────────────────
# user‐defined network
layer_sizes = [1, 2, 2, 1]

# build
N, layers, neighbors = build_network(layer_sizes)

# parameters (scalars or vector of length N)
m       = ones(N)
k_sink  = 0.2 .* ones(N)
c       = 0.05 .* ones(N)
k_cpl   = 0.5            # uniform coupling

# pack params
p = (m=m, k_sink=k_sink, k_cpl=k_cpl, c=c, neighbors=neighbors)

# random ICs in [-1,1]
x0 = 2rand(N).-1
v0 = 2rand(N).-1
u0 = vcat(x0,v0)

# time span & solve
tspan = (0.0, 10.0)
prob  = ODEProblem(spring_mass_ode!, u0, tspan, p)
sol   = solve(prob, Tsit5(), saveat=0:0.0333:10.0)

# compute energies
Elocal, Etotal = compute_energies(sol,p)
ts = sol.t

#  static plots
plot_all_nodes(ts, Elocal, Etotal)
plot_io_nodes(ts, Elocal, layers; layers=(1,length(layer_sizes)))
static_network_plot(layer_sizes)

#  animation
scene = animate_network(sol,p,layer_sizes)
display(scene)
