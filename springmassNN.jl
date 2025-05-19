using Plots
using Random

"""
    SpringMassNeuralNetwork

A Neural Network based on spring-mass physical system dynamics.
"""
mutable struct SpringMassNeuralNetwork
    layer_sizes::Vector{Int}             # Number of neurons in each layer
    num_layers::Int                      # Total number of layers
    weights::Vector{Matrix{Float64}}     # Connection weights between layers
    masses::Vector{Vector{Float64}}      # Masses of each neuron
    spring_constants::Vector{Matrix{Float64}} # Spring constants for connections
    sink_springs::Vector{Vector{Float64}}    # Sink spring constants (to ground)
    damping::Float64                     # Damping coefficient
    dt::Float64                          # Time step for simulation
    learning_rate::Float64               # Learning rate for weight updates
    
    """
        SpringMassNeuralNetwork(layer_sizes; damping=0.2, dt=0.01, learning_rate=0.05)
    
    Create a new Spring-Mass Neural Network with the specified layer architecture.
    
    Parameters:
    - `layer_sizes`: Vector of integers specifying the number of neurons in each layer
    - `damping`: Damping coefficient for the spring-mass system
    - `dt`: Time step for numerical integration
    - `learning_rate`: Learning rate for weight updates
    - `normalize`: Whether to normalize all parameters to [0,1] range
    
    Example:
    ```julia
    # Create a network with 2 inputs, 5 neurons in first hidden layer, 
    # 3 neurons in second hidden layer, and 1 output
    network = SpringMassNeuralNetwork([2, 5, 3, 1])
    ```
    """
    function SpringMassNeuralNetwork(layer_sizes::Vector{Int}; 
                                    damping::Float64=0.2, 
                                    dt::Float64=0.01, 
                                    learning_rate::Float64=0.05,
                                    mass_range::Tuple{Float64,Float64}=(0.8, 1.2),
                                    spring_constant_range::Tuple{Float64,Float64}=(0.8, 1.2),
                                    sink_spring_range::Tuple{Float64,Float64}=(0.5, 1.0),
                                    weight_range::Tuple{Float64,Float64}=(-0.5, 0.5),
                                    normalize::Bool=false,
                                    seed::Union{Int,Nothing}=nothing)
                                    
        # Set random seed if provided
        if seed !== nothing
            Random.seed!(seed)
        end
        
        num_layers = length(layer_sizes)
        
        if num_layers < 2
            error("Network must have at least 2 layers (input and output)")
        end
        
        # Initialize weights randomly
        weights = Vector{Matrix{Float64}}(undef, num_layers-1)
        for i in 1:(num_layers-1)
            weights[i] = weight_range[1] .+ (weight_range[2] - weight_range[1]) * 
                         rand(Float64, layer_sizes[i+1], layer_sizes[i])
        end
        
        # Initialize physical parameters - masses for each neuron
        masses = Vector{Vector{Float64}}(undef, num_layers)
        for l in 1:num_layers
            masses[l] = mass_range[1] .+ (mass_range[2] - mass_range[1]) * 
                        rand(Float64, layer_sizes[l])
        end
        
        # Spring constants for connections between layers
        spring_constants = Vector{Matrix{Float64}}(undef, num_layers-1)
        for i in 1:(num_layers-1)
            spring_constants[i] = spring_constant_range[1] .+ 
                                (spring_constant_range[2] - spring_constant_range[1]) * 
                                rand(Float64, layer_sizes[i+1], layer_sizes[i])
        end
        
        # Initialize sink springs (connections to ground)
        sink_springs = Vector{Vector{Float64}}(undef, num_layers)
        for l in 1:num_layers
            sink_springs[l] = sink_spring_range[1] .+ 
                             (sink_spring_range[2] - sink_spring_range[1]) * 
                             rand(Float64, layer_sizes[l])
        end
        
        # Normalize parameters to [0,1] if requested
        if normalize
            # Normalize weights to [-1,1]
            for i in 1:(num_layers-1)
                max_abs_weight = maximum(abs.(weights[i]))
                if max_abs_weight > 0
                    weights[i] ./= max_abs_weight
                end
            end
            
            # Normalize masses to [0,1]
            for l in 1:num_layers
                min_mass = minimum(masses[l])
                max_mass = maximum(masses[l])
                if max_mass > min_mass
                    masses[l] = (masses[l] .- min_mass) ./ (max_mass - min_mass)
                else
                    masses[l] .= 0.5
                end
            end
            
            # Normalize spring constants to [0,1]
            for i in 1:(num_layers-1)
                min_spring = minimum(spring_constants[i])
                max_spring = maximum(spring_constants[i])
                if max_spring > min_spring
                    spring_constants[i] = (spring_constants[i] .- min_spring) ./ 
                                         (max_spring - min_spring)
                else
                    spring_constants[i] .= 0.5
                end
            end
            
            # Normalize sink springs to [0,1]
            for l in 1:num_layers
                min_sink = minimum(sink_springs[l])
                max_sink = maximum(sink_springs[l])
                if max_sink > min_sink
                    sink_springs[l] = (sink_springs[l] .- min_sink) ./ (max_sink - min_sink)
                else
                    sink_springs[l] .= 0.5
                end
            end
        end
        
        new(layer_sizes, num_layers, weights, masses, spring_constants, sink_springs, damping, dt, learning_rate)
    end
end

"""
    print_network_info(network)

Print detailed information about the network architecture and parameters.
"""
function print_network_info(network::SpringMassNeuralNetwork)
    println("Spring-Mass Neural Network")
    println("==========================")
    println("Architecture: ", join(network.layer_sizes, " → "))
    println("Number of layers: ", network.num_layers)
    
    total_neurons = sum(network.layer_sizes)
    total_connections = sum(network.layer_sizes[1:end-1] .* network.layer_sizes[2:end])
    
    println("Total neurons: ", total_neurons)
    println("Total connections: ", total_connections)
    println()
    
    println("Physical Parameters:")
    println("  Damping coefficient: ", network.damping)
    println("  Time step (dt): ", network.dt)
    println("  Learning rate: ", network.learning_rate)
    println()
    
    println("Layer Details:")
    for l in 1:network.num_layers
        if l == 1
            layer_type = "Input"
        elseif l == network.num_layers
            layer_type = "Output"
        else
            layer_type = "Hidden"
        end
        
        println("  Layer $l ($layer_type): $(network.layer_sizes[l]) neurons")
        println("    Mass range: [$(round(minimum(network.masses[l]), digits=3)), $(round(maximum(network.masses[l]), digits=3))]")
        println("    Sink spring range: [$(round(minimum(network.sink_springs[l]), digits=3)), $(round(maximum(network.sink_springs[l]), digits=3))]")
        
        if l < network.num_layers
            println("    Connections to next layer: $(network.layer_sizes[l] * network.layer_sizes[l+1])")
            println("    Weight range: [$(round(minimum(network.weights[l]), digits=3)), $(round(maximum(network.weights[l]), digits=3))]")
            println("    Spring constant range: [$(round(minimum(network.spring_constants[l]), digits=3)), $(round(maximum(network.spring_constants[l]), digits=3))]")
        end
    end
end

"""
    visualize_network(network)

Create a visualization of the network architecture and connection weights,
including 3D representation of sink springs.
"""
function visualize_network(network::SpringMassNeuralNetwork)
    # Calculate node positions for visualization
    max_layer_size = maximum(network.layer_sizes)
    node_positions = Dict()
    
    # Calculate x and y coordinates for each node
    for l in 1:network.num_layers
        layer_size = network.layer_sizes[l]
        for i in 1:layer_size
            # Horizontal position (by layer)
            x = l
            # Vertical position (centered in layer)
            y = (max_layer_size - layer_size) / 2 + i
            node_positions[(l, i)] = (x, y)
        end
    end
    
    # Create plot with 3D-like floor
    p = plot(size=(800, 600), 
             xlim=(0.5, network.num_layers + 0.5), 
             ylim=(0, max_layer_size + 1.5),
             legend=false, 
             grid=false, 
             ticks=false, 
             framestyle=:none,
             title="Spring-Mass Neural Network Architecture")
    
    # Draw floor (for sink springs visualization)
    floor_y = 0.5
    plot!(p, [0.5, network.num_layers + 0.5], [floor_y, floor_y], 
          color=:gray, linewidth=2, alpha=0.5)
    
    # Draw sink springs (connections to floor)
    for l in 1:network.num_layers
        for i in 1:network.layer_sizes[l]
            pos = node_positions[(l, i)]
            
            # Calculate sink spring properties
            spring_constant = network.sink_springs[l][i]
            # Line thickness proportional to spring constant
            line_width = 1 + 2 * spring_constant
            
            # Draw sink spring as dashed line to floor
            plot!(p, [pos[1], pos[1]], [pos[2], floor_y], 
                  linestyle=:dash, 
                  color=:purple, 
                  linewidth=line_width, 
                  alpha=0.4)
        end
    end
    
    # Draw connections (edges) between layers
    for l in 1:(network.num_layers-1)
        for i in 1:network.layer_sizes[l]
            for j in 1:network.layer_sizes[l+1]
                src = node_positions[(l, i)]
                dst = node_positions[(l+1, j)]
                
                # Calculate color and width based on weight
                weight = network.weights[l][j, i]
                line_color = weight > 0 ? :blue : :red
                line_width = 1 + 3 * abs(weight) / max(0.1, maximum(abs.(network.weights[l])))
                
                # Draw line
                plot!(p, [src[1], dst[1]], [src[2], dst[2]], 
                     linewidth=line_width, 
                     color=line_color, 
                     alpha=0.6)
            end
        end
    end
    
    # Draw nodes
    for l in 1:network.num_layers
        # Determine node color based on layer type
        if l == 1
            node_color = :orange  # Input layer
        elseif l == network.num_layers
            node_color = :green   # Output layer
        else
            node_color = :lightblue  # Hidden layers
        end
        
        # Draw nodes for this layer
        for i in 1:network.layer_sizes[l]
            pos = node_positions[(l, i)]
            
            # Node size based on mass - larger mass = larger node
            mass = network.masses[l][i]
            node_size = 5 + 5 * (mass - minimum(network.masses[l])) / 
                        max(0.1, maximum(network.masses[l]) - minimum(network.masses[l]))
            
            # Draw node
            scatter!(p, [pos[1]], [pos[2]], 
                   markersize=node_size, 
                   color=node_color,
                   markerstrokewidth=1,
                   markerstrokecolor=:black)
        end
    end
    
    # Add layer labels
    for l in 1:network.num_layers
        if l == 1
            label = "Input"
        elseif l == network.num_layers
            label = "Output"
        else
            label = "Hidden $l"
        end
        
        annotate!(p, l, max_layer_size + 1, text(label, 10, :black))
    end
    
    # Add sink springs explanation
    annotate!(p, network.num_layers/2, floor_y/2, text("Sink Springs", 8, :purple))
    
    return p
end

"""
Create a test network and visualize it.
"""
function demo_network_creation()
    # Create a network with 2 inputs, two hidden layers (4 and 3 neurons), and 1 output
    network = SpringMassNeuralNetwork([2, 4, 3, 1], 
                                     damping=0.1, 
                                     dt=0.01, 
                                     learning_rate=0.05,
                                     mass_range=(0.5, 1.5),
                                     spring_constant_range=(0.7, 1.3),
                                     sink_spring_range=(0.3, 0.9),
                                     weight_range=(-1.0, 1.0),
                                     normalize=true,  # Normalize parameters to [0,1]
                                     seed=42)
    
    # Print network information
    print_network_info(network)
    
    # Visualize the network
    p = visualize_network(network)
    display(p)
    
    return network
end

# Run the demo if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    demo_network_creation()
end

# Create a network with 2 inputs, 4 and 3 hidden neurons, and 1 output
network = SpringMassNeuralNetwork([2, 4, 3, 1], 
                                normalize=true,  # Enable normalization
                                # Additional optional parameters:
                                damping=0.1,
                                dt=0.01,
                                learning_rate=0.05,
                                mass_range=(0.5, 1.5),
                                spring_constant_range=(0.7, 1.3),
                                sink_spring_range=(0.3, 0.9),
                                weight_range=(-1.0, 1.0),
                                seed=42)

# Visualize the network
visualize_network(network)