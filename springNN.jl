using DifferentialEquations
using LinearAlgebra
using CairoMakie
using GLMakie
using GeometryBasics: Point2f0, Point3f0, Sphere

# -------------------------------------------------------------------
# 1) Build arbitrary full‐bipartite inter‐layer neighbor list
function build_network(layer_sizes::Vector{Int})
    N = sum(layer_sizes)
    layers = Int[]
    for (l, n) in enumerate(layer_sizes)
        append!(layers, fill(l, n))
    end
    neighbors = [Int[] for _ in 1:N]
    offset = cumsum(vcat(0, layer_sizes))
    for l in 1:length(layer_sizes)-1
        for i in 1:layer_sizes[l], j in 1:layer_sizes[l+1]
            u = offset[l] + i
            v = offset[l+1] + j
            push!(neighbors[u], v)
            push!(neighbors[v], u)
        end
    end
    return N, layers, neighbors
end

# -------------------------------------------------------------------
# 2) The ODE system: u = [ x₁…x_N, v₁…v_N ]
function spring_mass_ode!(du, u, p, t)
    N = length(p.m)
    x = @view u[1:N]
    v = @view u[N+1:2N]
    dx = @view du[1:N]
    dv = @view du[N+1:2N]

    dx .= v
    @inbounds for i in 1:N
        f = -p.k_sink[i]*x[i] - p.c[i]*v[i]
        for j in p.neighbors[i]
            f += -p.k_cpl * (x[i] - x[j])
        end
        dv[i] = f / p.m[i]
    end
end

# -------------------------------------------------------------------
# 3) Compute local + total energies from the solution
function compute_energies(sol::ODESolution, p)
    N     = length(p.m)
    M     = Array(sol)         # (2N)×ntime
    xs    = M[1:N,   :]
    vs    = M[N+1:2N, :]
    ntime = length(sol.t)

    Ekin  = 0.5 .* p.m    .* (vs .^ 2)
    Esink = 0.5 .* p.k_sink .* (xs .^ 2)

    Ecpl = zeros(N, ntime)
    @inbounds for i in 1:N, ti in 1:ntime
        for j in p.neighbors[i]
            Ecpl[i,ti] += 0.5 * p.k_cpl * (xs[i,ti] - xs[j,ti])^2
        end
    end

    Elocal = Ekin .+ Esink .+ Ecpl
    Etotal = sum(Elocal, dims=1)[:]
    return Elocal, Etotal
end

# -------------------------------------------------------------------
# 4A) Static plot: all node energies + total
function plot_all_nodes(ts, Elocal, Etotal)
    fig = Figure(size=(800,400))
    ax  = Axis(fig[1,1], xlabel="t", ylabel="Energy", title="All Node Energies")
    lines!(ax, ts, Etotal, color=:black, label="Total", linewidth=2)
    for i in 1:size(Elocal,1)
        lines!(ax, ts, Elocal[i,:], linestyle=:dash, label="Node $i")
    end
    axislegend(ax)
    return fig
end

# -------------------------------------------------------------------
# 4B) Static plot: input‐layer vs output‐layer
function plot_io_nodes(ts, Elocal, layers; layers_idx=(1, maximum(layers)))
    fig = Figure(size=(800,300))
    ax1 = Axis(fig[1,1], xlabel="t", ylabel="Energy", title="Input Layer Energies")
    ax2 = Axis(fig[2,1], xlabel="t", ylabel="Energy", title="Output Layer Energies")
    inputs  = findall(x->x==layers_idx[1], layers)
    outputs = findall(x->x==layers_idx[2], layers)
    for i in inputs
        lines!(ax1, ts, Elocal[i,:], label="Node $i")
    end
    for j in outputs
        lines!(ax2, ts, Elocal[j,:], label="Node $j")
    end
    axislegend(ax1)
    axislegend(ax2)
    return fig
end

# -------------------------------------------------------------------
function static_network_plot(layer_sizes::Vector{Int})
    n_layers = length(layer_sizes)

    # 1) Compute node positions
    positions    = Point2f0[]
    layer_starts = Int[]
    idx = 1
    for (ℓ,n) in enumerate(layer_sizes)
        push!(layer_starts, idx)
        ys = n == 1  ? [0.0] :
             n == 2  ? [-0.5, 0.5] :
                       range(-1, 1; length=n)
        for y in ys
            push!(positions, Point2f0(ℓ*2, y))
            idx += 1
        end
    end

    # 2) Build arrow‐pairs between ℓ → ℓ+1
    arrow_pairs = Tuple{Point2f0,Point2f0}[]
    for ℓ in 1:n_layers-1
        s1, s2 = layer_starts[ℓ], layer_starts[ℓ+1]
        for i in 0:layer_sizes[ℓ]-1, j in 0:layer_sizes[ℓ+1]-1
            push!(arrow_pairs, (positions[s1+i], positions[s2+j]))
        end
    end

    # 3) Make figure & strip decorations
    fig = Figure(resolution=(600,400))
    ax  = Axis(fig[1,1])
    hidedecorations!(ax)  # hides ticks, tick labels, grid
    hidespines!(ax)       # hides the axis frame

    # 4) Draw grey arrows (with default arrowheads)
    for (p1,p2) in arrow_pairs
        arrows!(ax, [p1], [p2];
            color     = :gray,
            linewidth = 1,
        )
    end

    # 5) Draw nodes & labels
    for ℓ in 1:n_layers
        col = ℓ == 1           ? :blue       :  # input
              ℓ == n_layers    ? :red        :  # output
                                 :forestgreen   # hidden
        start = layer_starts[ℓ]
        for k in 0:layer_sizes[ℓ]-1
            idx = start + k
            p   = positions[idx]
            scatter!(ax, [p];
                color      = col,
                markersize = 30,
                strokewidth = 0,
            )
            text!(ax, string(idx);
                position = p,
                align    = (:center, :center),
                color    = :white,
                fontsize = 16,
            )
        end
    end

    return fig
end

# -------------------------------------------------------------------
# 5) GLMakie animation: spheres bob by local energy
function animate_network(sol::ODESolution, p, layer_sizes::Vector{Int})
    scene = Scene(size=(600,400), camera=campixel!)
    N, layers, neighbors = build_network(layer_sizes)

    positions = Vector{Point3f0}(undef, N)
    idx = 1
    for (l,n) in enumerate(layer_sizes)
        ys = n==1 ? [0.0] : range(0.0, -1.0, length=n)
        for i in 1:n
            positions[idx] = Point3f0(l, ys[i], 0.0)
            idx += 1
        end
    end

    spheres = [ mesh!(scene, Sphere(positions[i],0.08f0), color=:blue)
                for i in 1:N ]

    Elocal, _ = compute_energies(sol, p)
    ts = sol.t
    Emax = maximum(Elocal)

    @async for frame in 1:length(ts)
        for i in 1:N
            z = 0.2f0 * (Elocal[i,frame]/Emax)
            spheres[i].transformation.translation =
              Point3f0(positions[i][1], positions[i][2], z)
        end
        sleep(1/60)
    end

    return scene
end

# -------------------------------------------------------------------
# Main: ties everything together
function main()
    layer_sizes = [2,4,4,1]
    N, layers, neighbors = build_network(layer_sizes)

    m      = 1.0 .* ones(N)
    k_sink = 1.0 .* ones(N)
    c      = 0.0 .* ones(N)
    k_cpl  = 1.0

    p = (m=m, k_sink=k_sink, k_cpl=k_cpl, c=c, neighbors=neighbors)

    x0 = 2rand(N).-1
    v0 = 2rand(N).-1
    u0 = vcat(x0, v0)

    prob = ODEProblem(spring_mass_ode!, u0, (0.0, 50.0), p)
    sol  = solve(prob, Tsit5(), dt=0.01, saveat=0.01)

    Elocal, Etotal = compute_energies(sol, p)
    ts = sol.t

    CairoMakie.activate!()
    fig1 = plot_all_nodes(ts, Elocal, Etotal)
    display(fig1)   
    fig2 = plot_io_nodes(ts, Elocal, layers; layers_idx=(1,length(layer_sizes)))
    display(fig2)  
    fig3 = static_network_plot(layer_sizes)
    display(fig3)  

    GLMakie.activate!()
    scene = animate_network(sol, p, layer_sizes)
    display(scene) 
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
