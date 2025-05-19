using LinearAlgebra
using Plots
using Random

"""
    SpringMassNN

A neural network based on spring-mass physics.

# Fields
- `layers::Int`: Total number of layers including input and output
- `neurons::Vector{Int}`: Number of neurons per layer
- `x::Vector{Vector{Float64}}`: Position (activation) of each neuron
- `v::Vector{Vector{Float64}}`: Velocity of each neuron
- `a::Vector{Vector{Float64}}`: Acceleration of each neuron
- `m::Vector{Vector{Float64}}`: Mass of each neuron
- `K::Vector{Vector{Float64}}`: Sink spring constants
- `kPrev::Vector{Matrix{Float64}}`: Spring constants from previous layer
- `kNext::Vector{Matrix{Float64}}`: Spring constants to next layer
- `c::Vector{Vector{Float64}}`: Damping coefficients
"""
mutable struct SpringMassNN
    layers::Int
    neurons::Vector{Int}
    x::Vector{Vector{Float64}}
    v::Vector{Vector{Float64}}
    a::Vector{Vector{Float64}}
    m::Vector{Vector{Float64}}
    K::Vector{Vector{Float64}}
    kPrev::Vector{Matrix{Float64}}
    kNext::Vector{Matrix{Float64}}
    c::Vector{Vector{Float64}}
end

"""
    SpringMassNN(layers::Int, neurons::Vector{Int})

Initialize a spring-mass neural network with the given architecture.
"""
function SpringMassNN(layers::Int, neurons::Vector{Int})
    # Check validity
    if length(neurons) != layers
        throw(ArgumentError("The neurons vector must match the number of layers"))
    end
    
    # Initialize positions (activations) with small random values
    x = [0.1 * randn(neurons[l]) for l in 1:layers]
    
    # Initialize velocities to zero
    v = [zeros(neurons[l]) for l in 1:layers]
    
    # Initialize accelerations to zero
    a = [zeros(neurons[l]) for l in 1:layers]
    
    # Initialize masses (fixed for simplicity)
    m = [ones(neurons[l]) for l in 1:layers]
    
    # Initialize sink spring constants (biases)
    K = [rand(Float64, neurons[l]) * 0.4 .+ 0.1 for l in 1:layers]
    
    # Initialize coupling spring constants (weights)
    # From previous layer to current
    kPrev = Vector{Matrix{Float64}}(undef, layers-1)
    for l in 2:layers
        kPrev[l-1] = randn(neurons[l], neurons[l-1]) * 0.5
    end
    
    # From current layer to next
    kNext = Vector{Matrix{Float64}}(undef, layers-1)
    for l in 1:layers-1
        kNext[l] = randn(neurons[l], neurons[l+1]) * 0.5
    end
    
    # Initialize small damping coefficients (for stability)
    c = [fill(0.1, neurons[l]) for l in 1:layers]
    
    SpringMassNN(layers, neurons, x, v, a, m, K, kPrev, kNext, c)
end

"""
    kinetic_energy(nn::SpringMassNN, l::Int, i::Int)

Calculate kinetic energy for a node.
"""
function kinetic_energy(nn::SpringMassNN, l::Int, i::Int)
    0.5 * nn.m[l][i] * nn.v[l][i]^2
end

"""
    sink_energy(nn::SpringMassNN, l::Int, i::Int)

Calculate sink potential energy for a node.
"""
function sink_energy(nn::SpringMassNN, l::Int, i::Int)
    0.5 * nn.K[l][i] * nn.x[l][i]^2
end

"""
    prev_layer_energy(nn::SpringMassNN, l::Int, i::Int)

Calculate coupling energy from previous layer.
"""
function prev_layer_energy(nn::SpringMassNN, l::Int, i::Int)
    if l > 1
        energy = 0.0
        for n in 1:nn.neurons[l-1]
            energy += 0.5 * nn.kPrev[l-1][i, n] * (nn.x[l][i] - nn.x[l-1][n])^2
        end
        return energy
    else
        return 0.0
    end
end

"""
    next_layer_energy(nn::SpringMassNN, l::Int, i::Int)

Calculate coupling energy to next layer.
"""
function next_layer_energy(nn::SpringMassNN, l::Int, i::Int)
    if l < nn.layers
        energy = 0.0
        for m in 1:nn.neurons[l+1]
            energy += 0.5 * nn.kNext[l][i, m] * (nn.x[l+1][m] - nn.x[l][i])^2
        end
        return energy
    else
        return 0.0
    end
end

"""
    node_energy(nn::SpringMassNN, l::Int, i::Int)

Calculate total energy for a node.
"""
function node_energy(nn::SpringMassNN, l::Int, i::Int)
    kinetic_energy(nn, l, i) + sink_energy(nn, l, i) + 
    prev_layer_energy(nn, l, i) + next_layer_energy(nn, l, i)
end

"""
    network_energy(nn::SpringMassNN)

Calculate total network energy.
"""
function network_energy(nn::SpringMassNN)
    energy = 0.0
    for l in 1:nn.layers
        for i in 1:nn.neurons[l]
            energy += node_energy(nn, l, i)
        end
    end
    energy
end

"""
    calculate_forces!(nn::SpringMassNN)

Calculate forces on each node and return a vector of vectors.
Forces are calculated as the negative gradient of energy:
F = -∇E = -K*x - k*(x_i - x_j) - c*v
"""
function calculate_forces!(nn::SpringMassNN)
    forces = [zeros(nn.neurons[l]) for l in 1:nn.layers]
    
    for l in 1:nn.layers
        for i in 1:nn.neurons[l]
            # Force from sink potential: F = -K*x
            forces[l][i] = -nn.K[l][i] * nn.x[l][i]
            
            # Force from previous layer coupling: F = -k*(x_i - x_{i-1})
            if l > 1
                for n in 1:nn.neurons[l-1]
                    forces[l][i] -= nn.kPrev[l-1][i, n] * (nn.x[l][i] - nn.x[l-1][n])
                end
            end
            
            # Force from next layer coupling: F = -k*(x_i - x_{i+1})
            if l < nn.layers
                for m in 1:nn.neurons[l+1]
                    forces[l][i] -= nn.kNext[l][i, m] * (nn.x[l][i] - nn.x[l+1][m])
                end
            end
            
            # Damping force: F = -c*v
            forces[l][i] -= nn.c[l][i] * nn.v[l][i]
        end
    end
    
    return forces
end

"""
    calculate_accelerations!(nn::SpringMassNN, forces)

Calculate accelerations from forces using a = F/m.
"""
function calculate_accelerations!(nn::SpringMassNN, forces)
    accelerations = [zeros(nn.neurons[l]) for l in 1:nn.layers]
    
    for l in 1:nn.layers
        for i in 1:nn.neurons[l]
            accelerations[l][i] = forces[l][i] / nn.m[l][i]
        end
    end
    
    return accelerations
end

"""
    velocity_verlet_step!(nn::SpringMassNN, dt::Float64)

Perform one step of velocity Verlet integration using:
1. v(t+dt/2) = v(t) + (dt/2)*a(t)
2. x(t+dt) = x(t) + dt*v(t+dt/2)
3. a(t+dt) = F(x(t+dt))/m
4. v(t+dt) = v(t+dt/2) + (dt/2)*a(t+dt)
"""
function velocity_verlet_step!(nn::SpringMassNN, dt::Float64)
    # First half of velocity update: v(t+dt/2) = v(t) + (dt/2)*a(t)
    for l in 1:nn.layers
        for i in 1:nn.neurons[l]
            # Skip input layer nodes (they're fixed)
            if l == 1
                continue
            end
            nn.v[l][i] += 0.5 * nn.a[l][i] * dt
        end
    end
    
    # Position update: x(t+dt) = x(t) + dt*v(t+dt/2)
    for l in 1:nn.layers
        for i in 1:nn.neurons[l]
            # Skip input layer nodes (they're fixed)
            if l == 1
                continue
            end
            nn.x[l][i] += nn.v[l][i] * dt
        end
    end
    
    # Calculate new forces and accelerations
    forces = calculate_forces!(nn)
    new_accel = calculate_accelerations!(nn, forces)
    
    # Second half of velocity update: v(t+dt) = v(t+dt/2) + (dt/2)*a(t+dt)
    for l in 1:nn.layers
        for i in 1:nn.neurons[l]
            # Skip input layer nodes (they're fixed)
            if l == 1
                continue
            end
            nn.v[l][i] += 0.5 * new_accel[l][i] * dt
            nn.a[l][i] = new_accel[l][i]
        end
    end
end

"""
    update_spring_constants!(nn::SpringMassNN, learning_rate::Float64)

Update spring constants based on energy gradients:
∂E/∂k = -1/2 * (x_i - x_j)^2 for spring constant k connecting nodes i and j
Learning rule: k_new = k_old - learning_rate * ∂E/∂k
"""
function update_spring_constants!(nn::SpringMassNN, learning_rate::Float64)
    # Update sink spring constants
    for l in 1:nn.layers
        for i in 1:nn.neurons[l]
            # Gradient of energy with respect to sink spring constant
            delta_K = -learning_rate * 0.5 * nn.x[l][i]^2
            nn.K[l][i] -= delta_K
            
            # Ensure spring constants remain positive for stability
            if nn.K[l][i] < 0.01
                nn.K[l][i] = 0.01
            end
        end
    end
    
    # Update coupling spring constants from previous layer
    for l in 2:nn.layers
        for i in 1:nn.neurons[l]
            for n in 1:nn.neurons[l-1]
                # Gradient of energy with respect to previous layer coupling
                delta_prev = -learning_rate * 0.5 * (nn.x[l][i] - nn.x[l-1][n])^2
                nn.kPrev[l-1][i, n] -= delta_prev
            end
        end
    end
    
    # Update coupling spring constants to next layer
    for l in 1:nn.layers-1
        for i in 1:nn.neurons[l]
            for m in 1:nn.neurons[l+1]
                # Gradient of energy with respect to next layer coupling
                delta_next = -learning_rate * 0.5 * (nn.x[l+1][m] - nn.x[l][i])^2
                nn.kNext[l][i, m] -= delta_next
            end
        end
    end
end

"""
    set_input!(nn::SpringMassNN, input_values::Vector{Float64})

Set input layer values and fix velocities.
"""
function set_input!(nn::SpringMassNN, input_values::Vector{Float64})
    if length(input_values) != nn.neurons[1]
        throw(ArgumentError("Input length doesn't match network input layer size"))
    end
    
    for i in 1:nn.neurons[1]
        nn.x[1][i] = input_values[i]
        nn.v[1][i] = 0.0  # Fix velocity of input nodes
        nn.a[1][i] = 0.0  # Fix acceleration of input nodes
    end
end

"""
    set_target!(nn::SpringMassNN, target_values::Vector{Float64})

Set target values for output layer.
"""
function set_target!(nn::SpringMassNN, target_values::Vector{Float64})
    if length(target_values) != nn.neurons[end]
        throw(ArgumentError("Target length doesn't match network output layer size"))
    end
    
    for i in 1:nn.neurons[end]
        # Strengthen the sink constant to pull toward target
        nn.K[end][i] = 10.0
        nn.x[end][i] = target_values[i]
    end
end

"""
    predict(nn::SpringMassNN, input::Vector{Float64}, steps::Int=200, dt::Float64=0.01)

Run the network forward and return the output layer activations.
"""
function predict(nn::SpringMassNN, input::Vector{Float64}, steps::Int=200, dt::Float64=0.01)
    # Set input
    set_input!(nn, input)
    
    # Reset velocities and accelerations for non-input nodes
    for l in 2:nn.layers
        for i in 1:nn.neurons[l]
            nn.v[l][i] = 0.0
            nn.a[l][i] = 0.0
        end
    end
    
    # Let system evolve
    for _ in 1:steps
        velocity_verlet_step!(nn, dt)
    end
    
    # Return output layer activations
    return copy(nn.x[end])
end

"""
    train!(nn::SpringMassNN, inputs::Vector{Vector{Float64}}, 
           targets::Vector{Vector{Float64}}, epochs::Int=100,
           relaxation_steps::Int=200, dt::Float64=0.01, learning_rate::Float64=0.005)

Train the spring-mass neural network on the given dataset.
"""
function train!(nn::SpringMassNN, inputs::Vector{Vector{Float64}}, 
                targets::Vector{Vector{Float64}}, epochs::Int=100,
                relaxation_steps::Int=200, dt::Float64=0.01, learning_rate::Float64=0.005)
    
    energy_history = Float64[]
    errors_history = Float64[]
    
    for epoch in 1:epochs
        epoch_error = 0.0
        
        # Loop through training examples
        for example in 1:length(inputs)
            # Set input
            set_input!(nn, inputs[example])
            
            # Reset velocities and accelerations for non-input nodes
            for l in 2:nn.layers
                for i in 1:nn.neurons[l]
                    nn.v[l][i] = 0.0
                    nn.a[l][i] = 0.0
                end
            end
            
            # Let system evolve
            for step in 1:relaxation_steps
                velocity_verlet_step!(nn, dt)
            end
            
            # Calculate error
            error = sum((nn.x[end][i] - targets[example][i])^2 for i in 1:nn.neurons[end])
            epoch_error += error
            
            # Record energy
            push!(energy_history, network_energy(nn))
            
            # Set target and update spring constants
            set_target!(nn, targets[example])
            update_spring_constants!(nn, learning_rate)
        end
        
        # Record error
        push!(errors_history, epoch_error)
        
        # Print progress
        if epoch % 10 == 0
            println("Epoch $epoch: Error = $epoch_error")
        end
    end
    
    return errors_history, energy_history
end

"""
    test_network(nn::SpringMassNN, inputs::Vector{Vector{Float64}}, 
                targets::Vector{Vector{Float64}})

Test the network on the given dataset and print results.
"""
function test_network(nn::SpringMassNN, inputs::Vector{Vector{Float64}}, 
                     targets::Vector{Vector{Float64}})
    println("Testing Network Results:")
    println("-----------------------")
    
    for i in 1:length(inputs)
        prediction = predict(nn, inputs[i])
        println("Input: $(inputs[i]) → Output: $prediction (Expected: $(targets[i]))")
    end
end

"""
    visualize_network(nn::SpringMassNN)

Visualize the network structure using Plots.jl.
"""
function visualize_network(nn::SpringMassNN)
    # Create plot
    p = plot(size=(800, 600), legend=false, title="Spring-Mass Neural Network")
    
    # Calculate positions for layers
    layer_x = [i for i in 1:nn.layers]
    
    # Plot nodes
    for l in 1:nn.layers
        for i in 1:nn.neurons[l]
            # Calculate y position to spread nodes vertically
            y_pos = (i - (nn.neurons[l] + 1)/2) * 2
            
            # Node color by layer
            node_color = l == 1 ? :red : (l == nn.layers ? :blue : :green)
            
            # Plot node
            scatter!(p, [layer_x[l]], [y_pos], markersize=10, color=node_color)
            
            # Plot connections to previous layer
            if l > 1
                for n in 1:nn.neurons[l-1]
                    prev_y = (n - (nn.neurons[l-1] + 1)/2) * 2
                    
                    # Calculate line thickness based on spring constant
                    line_width = abs(nn.kPrev[l-1][i, n])
                    
                    # Plot connection
                    plot!(p, [layer_x[l-1], layer_x[l]], [prev_y, y_pos], 
                          linewidth=line_width*3, color=:gray, alpha=0.6)
                end
            end
        end
    end
    
    # Set axis labels and limits
    xlabel!(p, "Layer")
    ylabel!(p, "Node")
    xlims!(p, 0.5, nn.layers + 0.5)
    
    return p
end

# Example usage for XOR problem
function run_xor_example()
    # Create network
    nn = SpringMassNN(3, [2, 3, 1])
    
    # Define XOR training data
    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    targets = [[0.0], [1.0], [1.0], [0.0]]
    
    # Train network
    errors, energies = train!(nn, inputs, targets, 100, 200, 0.01, 0.005)
    
    # Test network
    test_network(nn, inputs, targets)
    
    # Visualize training progress and network
    p1 = plot(errors, xlabel="Epoch", ylabel="Error", title="Training Error", legend=false)
    p2 = plot(energies, xlabel="Example", ylabel="Energy", title="Network Energy", legend=false)
    p3 = visualize_network(nn)
    
    # Display plots
    plt = plot(p1, p2, p3, layout=(3,1), size=(800, 900))
    display(plt)
    
    return nn, plt
end

# Print equations reference
println("""
Spring-Mass Neural Network - Equations:
----------------------------------------
1. Energy Components:
   - Kinetic Energy: E_kin(i) = (1/2)m_i v_i^2
   - Sink Potential: E_sink(i) = (1/2)K_i x_i^2
   - Previous Layer Coupling: E_prev(i) = Sum[(1/2)k_{prev,i,j}(x_i - x_{j})^2]
   - Next Layer Coupling: E_next(i) = Sum[(1/2)k_{next,i,j}(x_j - x_i)^2]

2. Force Equation (for each node):
   F_i = -∇E_i = -K_i x_i - Sum[k_{prev,i,j}(x_i - x_j)] - Sum[k_{next,i,j}(x_i - x_j)] - c_i v_i

3. Acceleration:
   a_i = F_i / m_i

4. Velocity Verlet Integration:
   v_i(t+dt/2) = v_i(t) + (dt/2)a_i(t)
   x_i(t+dt) = x_i(t) + dt*v_i(t+dt/2)
   a_i(t+dt) = F_i(x(t+dt)) / m_i
   v_i(t+dt) = v_i(t+dt/2) + (dt/2)a_i(t+dt)

5. Learning Rule (Spring Constant Update):
   ∆k = -η * ∂E/∂k = -η * (1/2)(x_i - x_j)^2
   k_new = k_old - ∆k
""")

# Run the example
nn, plots = run_xor_example()