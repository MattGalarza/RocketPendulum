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
    
    # Initialize masses (could be learned but fixed for simplicity)
    m = [ones(neurons[l]) for l in 1:layers]
    
    # Initialize sink spring constants (biases)
    K = [rand(Float64, neurons[l]) * 0.4 .+ 0.1 for l in 1:layers]
    
    # Initialize coupling spring constants (weights)
    # From previous layer to current
    kPrev = [randn(neurons[l], neurons[l-1]) * 0.5 for l in 2:layers]
    
    # From current layer to next
    kNext = [randn(neurons[l], neurons[l+1]) * 0.5 for l in 1:layers-1]
    
    # Initialize small damping coefficients
    c = [fill(0.05, neurons[l]) for l in 1:layers]
    
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
"""
function calculate_forces!(nn::SpringMassNN)
    forces = [zeros(nn.neurons[l]) for l in 1:nn.layers]
    
    for l in 1:nn.layers
        for i in 1:nn.neurons[l]
            # Force from sink potential
            forces[l][i] = -nn.K[l][i] * nn.x[l][i]
            
            # Force from previous layer coupling
            if l > 1
                for n in 1:nn.neurons[l-1]
                    forces[l][i] -= nn.kPrev[l-1][i, n] * (nn.x[l][i] - nn.x[l-1][n])
                end
            end
            
            # Force from next layer coupling
            if l < nn.layers
                for m in 1:nn.neurons[l+1]
                    forces[l][i] -= nn.kNext[l][i, m] * (nn.x[l][i] - nn.x[l+1][m])
                end
            end
            
            # Damping force
            forces[l][i] -= nn.c[l][i] * nn.v[l][i]
        end
    end
    
    forces
end

"""
    calculate_accelerations!(nn::SpringMassNN, forces)

Calculate accelerations from forces.
"""
function calculate_accelerations!(nn::SpringMassNN, forces)
    accelerations = [zeros(nn.neurons[l]) for l in 1:nn.layers]
    
    for l in 1:nn.layers
        for i in 1:nn.neurons[l]
            accelerations[l][i] = forces[l][i] / nn.m[l][i]
        end
    end
    
    accelerations
end

"""
    velocity_verlet_step!(nn::SpringMassNN, dt::Float64)

Perform one step of velocity Verlet integration.
"""
function velocity_verlet_step!(nn::SpringMassNN, dt::Float64)
    # First half of velocity update
    for l in 1:nn.layers
        for i in 1:nn.neurons[l]
            nn.v[l][i] += 0.5 * nn.a[l][i] * dt
        end
    end
    
    # Position update
    for l in 1:nn.layers
        for i in 1:nn.neurons[l]
            nn.x[l][i] += nn.v[l][i] * dt
        end
    end
    
    # Calculate new forces and accelerations
    forces = calculate_forces!(nn)
    new_accel = calculate_accelerations!(nn, forces)
    
    # Second half of velocity update
    for l in 1:nn.layers
        for i in 1:nn.neurons[l]
            nn.v[l][i] += 0.5 * new_accel[l][i] * dt
            nn.a[l][i] = new_accel[l][i]
        end
    end
end

"""
    update_spring_constants!(nn::SpringMassNN, learning_rate::Float64)

Update spring constants based on energy gradients.
"""
function update_spring_constants!(nn::SpringMassNN, learning_rate::Float64)
    # Update sink spring constants
    for l in 1:nn.layers
        for i in 1:nn.neurons[l]
            delta_K = -learning_rate * nn.x[l][i]^2
            nn.K[l][i] += delta_K
        end
    end
    
    # Update coupling spring constants from previous layer
    for l in 2:nn.layers
        for i in 1:nn.neurons[l]
            for n in 1:nn.neurons[l-1]
                delta_prev = -learning_rate * (nn.x[l][i] - nn.x[l-1][n])^2
                nn.kPrev[l-1][i, n] += delta_prev
            end
        end
    end
    
    # Update coupling spring constants to next layer
    for l in 1:nn.layers-1
        for i in 1:nn.neurons[l]
            for m in 1:nn.neurons[l+1]
                delta_next = -learning_rate * (nn.x[l+1][m] - nn.x[l][i])^2
                nn.kNext[l][i, m] += delta_next
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
    predict(nn::SpringMassNN, input::Vector{Float64}, steps::Int=500, dt::Float64=0.01)

Run the network forward and return the output layer activations.
"""
function predict(nn::SpringMassNN, input::Vector{Float64}, steps::Int=500, dt::Float64=0.01)
    # Set input
    set_input!(nn, input)
    
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
           relaxation_steps::Int=500, dt::Float64=0.01, learning_rate::Float64=0.001)

Train the spring-mass neural network on the given dataset.
"""
function train!(nn::SpringMassNN, inputs::Vector{Vector{Float64}}, 
                targets::Vector{Vector{Float64}}, epochs::Int=100,
                relaxation_steps::Int=500, dt::Float64=0.01, learning_rate::Float64=0.001)
    
    energy_history = Float64[]
    errors_history = Float64[]
    
    for epoch in 1:epochs
        epoch_error = 0.0
        
        # Loop through training examples
        for example in 1:length(inputs)
            # Set input
            set_input!(nn, inputs[example])
            
            # Let system evolve
            for step in 1:relaxation_steps
                velocity_verlet_step!(nn, dt)
                
                # Record energy on last step
                if step == relaxation_steps
                    push!(energy_history, network_energy(nn))
                end
            end
            
            # Calculate error
            error = sum((nn.x[end][i] - targets[example][i])^2 for i in 1:nn.neurons[end])
            epoch_error += error
            
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

# Example usage for XOR problem
function run_xor_example()
    # Create network
    nn = SpringMassNN(3, [2, 3, 1])
    
    # Define XOR training data
    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    targets = [[0.0], [1.0], [1.0], [0.0]]
    
    # Train network
    errors, energies = train!(nn, inputs, targets, 100, 500, 0.01, 0.001)
    
    # Test network
    test_network(nn, inputs, targets)
    
    # Visualize training progress
    p1 = plot(errors, xlabel="Epoch", ylabel="Error", title="Training Error", legend=false)
    p2 = plot(energies, xlabel="Example", ylabel="Energy", title="Network Energy", legend=false)
    plot(p1, p2, layout=(2,1), size=(800, 600))
    
    return nn, p1, p2
end

# Run the example
nn, error_plot, energy_plot = run_xor_example()