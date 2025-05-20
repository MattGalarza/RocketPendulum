using DifferentialEquations, OrdinaryDiffEq
using LinearAlgebra, StaticArrays, Statistics
using CairoMakie, GLMakie, MakieCore, GeometryBasics, Colors, Observables 

# ------------------------- Build spring network -------------------------
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

# ------------------------- Build network ODE -------------------------
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

# ------------------------- Compute local and total energies -------------------------
function compute_energies(sol::ODESolution, p)
    N = length(p.m)
    M = Array(sol)       
    xs = M[1:N,:]
    vs = M[N+1:2N,:]
    ntime = length(sol.t)

    # Local node energies (kinetic and sink)
    Ekin  = 0.5 .* p.m .* (vs .^ 2)
    Esink = 0.5 .* p.k_sink .* (xs .^ 2)

    # Local coupling energies
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

# ------------------------- Plotting + Visualization -------------------------
# Static plot (all nodes)
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

# Static plot (input/output nodes)
function plot_io_nodes(ts, Elocal, layers; layers_idx=(1, maximum(layers)))
    fig = Figure(size=(800,400))
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

# Network Schemcatic
function static_network_plot(layer_sizes::Vector{Int})
    n_layers = length(layer_sizes)
    xs = range(0f0, 1f0; length=n_layers)
    positions = Point2f0[]
    layer_starts = Int[]
    idx = 1
    for (l,n) in enumerate(layer_sizes)
        push!(layer_starts, idx)
        ys = n == 1 ? [0.5f0] : n == 2 ? [0.3f0,0.7f0] : range(0.1f0, 0.9f0; length=n)
        for y in ys
            push!(positions, Point2f0(xs[l], y))
            idx += 1
        end
    end

    # prepare the layers
    pairs = Tuple{Point2f0,Point2f0}[]
    for l in 1:n_layers-1
        s1, s2 = layer_starts[l], layer_starts[l+1]
        for i in 0:layer_sizes[l]-1, j in 0:layer_sizes[l+1]-1
            p1 = positions[s1 + i]
            p2 = positions[s2 + j]
            push!(pairs, (p1, p2))
        end
    end

    # Remove figure axes and grid
    fig = Figure(size=(800,300))
    ax  = Axis(fig[1,1];
        xgridvisible=false,
        ygridvisible=false,
        xticksvisible=false,
        yticksvisible=false,
        xticklabelsvisible=false,
        yticklabelsvisible=false,
        leftspinevisible=false,
        rightspinevisible=false,
        topspinevisible=false,
        bottomspinevisible=false
    )
    limits!(ax, -0.05, 1.05, -0.05, 1.05)

    # Connect nodes with lines
    for (p1, p2) in pairs
        lines!(ax, [p1, p2]; color=:gray, linewidth=1.2)
    end

    # Include nodes and labels
    for l in 1:n_layers
        col = l==1 ? :blue : l==n_layers ? :red : :forestgreen
        start = layer_starts[l]
        for k in 0:layer_sizes[l]-1
            i = start + k
            p = positions[i]
            scatter!(ax, [p];
                color = col,
                markersize = 28,
                strokewidth = 0,
            )
            text!(ax, string(i);
                position = p,
                align = (:center, :center),
                color = :white,
                fontsize = 16,
            )
        end
    end
    return fig
end

# Visualization of energy network
function animate_network3d(sol, p, layer_sizes::Vector{Int})
    N, layers, neighbors = build_network(layer_sizes)
    xs = Float32.(range(0f0, 2f0*(length(layer_sizes)-1), length=length(layer_sizes)))
    base_positions = Point3f0[]
    for (l,n) in enumerate(layer_sizes)
        ys = n==1 ? [0f0] : Float32.(range(-1.5,1.5,length=n))
        for y in ys
            push!(base_positions, Point3f0(xs[l], y, 0f0))
        end
    end

    conns = Tuple{Int,Int}[]
    for i in 1:N, j in neighbors[i]
        i<j && push!(conns,(i,j))
    end
    make_segments(pts) = begin
        seg = Point3f0[]
        for (i,j) in conns
            push!(seg, pts[i], pts[j])
        end
        seg
    end

    # Compute energies
    Elocal, _ = compute_energies(sol, p)
    Emax = maximum(Elocal)
    nframes = length(sol.t)

    # Node colors
    node_colors = [layer==1 ? :royalblue :
                    layer==maximum(layers) ? :firebrick : :forestgreen
                    for layer in layers
    ]

    # Build scene
    scene = Scene(size=(800,600), backgroundcolor=:white)
    positionsObs = Observable(copy(base_positions))
    meshscatter!(
        scene, positionsObs;
        markersize = 0.3f0,
        color = node_colors,
        #shading = true,
        transparency = false
    )
    segObs = Observable(make_segments(base_positions))
    lines!(scene, segObs; color=:gray, linewidth=1)

    x_mid = sum(p[1] for p in base_positions)/N
    y_mid = sum(p[2] for p in base_positions)/N
    cam3d!(scene,
        lookat = Point3f0(x_mid, y_mid, 0f0),
        eyeposition = Point3f0(-6, -6, 4)
    )

    # Animation loop
    @async for frame in 1:nframes
        new_pos = Point3f0[]
        for i in 1:N
            z = Emax==0f0 ? 0f0 : Float32( Elocal[i,frame] / Emax )
            bp = base_positions[i]
            push!(new_pos, Point3f0(bp[1], bp[2], z))
        end
        # update points and springs
        positionsObs[] = new_pos
        segObs[] = make_segments(new_pos)
        sleep(1/60)
    end
    return scene
end

# ------------------------- Generate network + solve ODE -------------------------
# Main: ties everything together
function main()
    layer_sizes = [2,4,4,1]
    N, layers, neighbors = build_network(layer_sizes)

    m = 1.0 .* ones(N)
    k_sink = 1.0 .* ones(N)
    c = 0.0 .* ones(N)
    k_cpl = 1.0

    p = (m=m, k_sink=k_sink, k_cpl=k_cpl, c=c, neighbors=neighbors)

    x0 = 2rand(N).-1
    v0 = 2rand(N).-1
    u0 = vcat(x0, v0)

    prob = ODEProblem(spring_mass_ode!, u0, (0.0, 10.0), p)
    sol = solve(prob, Tsit5(), dt=0.01, saveat=0.01)

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
    scene = animate_network3d(sol, p, layer_sizes)
    display(scene)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end