using LinearAlgebra
using Plots
using Random
using Printf

"""
    SpringMassNN

A simplified spring-mass neural network implementation with robust numerical handling.
"""
mutable struct SpringMassNN
    layers::Int                      # Number of layers
    neurons::Vector{Int}             # Neurons per layer
    x::Vector{Vector{Float64}}       # Positions (activations)
    v::Vector{Vector{Float64}}       # Velocities
    a::Vector{Vector{Float64}}       # Accelerations
    m::Vector{Vector{Float64}}       # Masses
    K::Vector{Vector{Float64}}       # Sink spring constants (biases)
    w::Vector{Matrix{Float64}}       # Weights (spring constants)
    c::Vector{Vector{Float64}}       # Damping coefficients
    
    # Constructor
    function SpringMassNN(layers::Int, neurons::Vector{Int})
        # Validate inputs
        if length(neurons) != layers
            throw(ArgumentError("Neurons vector must match number of layers"))
        end
        
        # Initialize positions
        x = [zeros(neurons[l]) for l in 1:layers]
        
        # Initialize velocities and accelerations
        v = [zeros(neurons[l]) for l in 1:layers]
        a = [zeros(neurons[l]) for l in 1:layers]
        
        # Initialize masses (fixed at 1.0)
        m = [ones(neurons[l]) for l in 1:layers]
        
        # Initialize sink spring constants
        K = [fill(0.2, neurons[l]) for l in 1:layers]
        
        # Initialize weights with improved scaling
        w = Vector{Matrix{Float64}}(undef, layers-1)
        for l in 2:layers
            w[l-1] = 0.2 * randn(neurons[l], neurons[l-1])
        end
        
        # Initialize damping
        c = [fill(0.2, neurons[l]) for l in 1:layers]
        
        new(layers, neurons, x, v, a, m, K, w, c)
    end
end

"""
    network_energy(nn::SpringMassNN)

Calculate the total energy of the network.
"""
function network_energy(nn::SpringMassNN)
    energy = 0.0
    
    # Kinetic energy
    for l in 1:nn.layers
        for i in 1:nn.neurons[l]
            energy += 0.5 * nn.m[l][i] * nn.v[l][i]^2
        end
    end
    
    # Sink potential energy
    for l in 1:nn.layers
        for i in 1:nn.neurons[l]
            energy += 0.5 * nn.K[l][i] * nn.x[l][i]^2
        end
    end
    
    # Coupling energy - consistent with how forces are calculated
    for l in 2:nn.layers
        for i in 1:nn.neurons[l]
            for j in 1:nn.neurons[l-1]
                energy += 0.5 * nn.w[l-1][i, j] * (nn.x[l][i] - nn.x[l-1][j])^2
            end
        end
    end
    
    return energy
end

"""
    calculate_forces!(nn::SpringMassNN)

Calculate forces for each node based on energy gradients.
"""
function calculate_forces!(nn::SpringMassNN)
    forces = [zeros(nn.neurons[l]) for l in 1:nn.layers]
    
    # Calculate forces for each node
    for l in 1:nn.layers
        for i in 1:nn.neurons[l]
            # Sink force
            forces[l][i] = -nn.K[l][i] * nn.x[l][i]
            
            # Coupling forces - from previous layer
            if l > 1
                for j in 1:nn.neurons[l-1]
                    forces[l][i] -= nn.w[l-1][i, j] * (nn.x[l][i] - nn.x[l-1][j])
                end
            end
            
            # Coupling forces - to next layer
            if l < nn.layers
                for j in 1:nn.neurons[l+1]
                    # Note: Fixed indexing here - weights between layer l and l+1 are stored in nn.w[l]
                    forces[l][i] -= nn.w[l][j, i] * (nn.x[l][i] - nn.x[l+1][j])
                end
            end
            
            # Damping force
            forces[l][i] -= nn.c[l][i] * nn.v[l][i]
        end
    end
    
    return forces
end

"""
    integration_step!(nn::SpringMassNN, dt::Float64)

Perform one step of velocity Verlet integration.
"""
function integration_step!(nn::SpringMassNN, dt::Float64)
    # First half of velocity update
    for l in 2:nn.layers  # Skip input layer
        for i in 1:nn.neurons[l]
            nn.v[l][i] += 0.5 * nn.a[l][i] * dt
        end
    end
    
    # Position update
    for l in 2:nn.layers  # Skip input layer
        for i in 1:nn.neurons[l]
            nn.x[l][i] += nn.v[l][i] * dt
        end
    end
    
    # Recalculate forces
    forces = calculate_forces!(nn)
    
    # Second half of velocity update
    for l in 2:nn.layers  # Skip input layer
        for i in 1:nn.neurons[l]
            nn.a[l][i] = forces[l][i] / nn.m[l][i]
            nn.v[l][i] += 0.5 * nn.a[l][i] * dt
        end
    end
end

"""
    set_input!(nn::SpringMassNN, input::Vector{Float64})

Set the input layer values.
"""
function set_input!(nn::SpringMassNN, input::Vector{Float64})
    if length(input) != nn.neurons[1]
        throw(ArgumentError("Input length doesn't match network input size"))
    end
    
    for i in 1:nn.neurons[1]
        nn.x[1][i] = input[i]
        nn.v[1][i] = 0.0  # Fix input velocities
        nn.a[1][i] = 0.0  # Fix input accelerations
    end
end

"""
    set_target!(nn::SpringMassNN, target::Vector{Float64})

Set the target values for the output layer.
"""
function set_target!(nn::SpringMassNN, target::Vector{Float64})
    if length(target) != nn.neurons[end]
        throw(ArgumentError("Target length doesn't match network output size"))
    end
    
    for i in 1:nn.neurons[end]
        # Use large sink spring to pull toward target
        nn.K[end][i] = 10.0
        nn.x[end][i] = target[i]
    end
end

"""
    predict(nn::SpringMassNN, input::Vector{Float64}, steps::Int=100, dt::Float64=0.01)

Run the network forward and return the output.
"""
function predict(nn::SpringMassNN, input::Vector{Float64}, steps::Int=100, dt::Float64=0.01)
    set_input!(nn, input)
    
    # Reset dynamics
    for l in 2:nn.layers
        for i in 1:nn.neurons[l]
            nn.v[l][i] = 0.0
            nn.a[l][i] = 0.0
        end
    end
    
    # Run dynamics
    for _ in 1:steps
        integration_step!(nn, dt)
    end
    
    return copy(nn.x[end])
end

"""
    learn_step!(nn::SpringMassNN, learning_rate::Float64)

Update spring constants based on energy gradients.
"""
function learn_step!(nn::SpringMassNN, learning_rate::Float64)
    # Update coupling spring constants
    for l in 2:nn.layers
        for i in 1:nn.neurons[l]
            for j in 1:nn.neurons[l-1]
                # Gradient of energy w.r.t. spring constant
                delta_w = learning_rate * (nn.x[l][i] - nn.x[l-1][j])^2 / 2
                
                # Update spring constant
                nn.w[l-1][i, j] -= delta_w
                
                # Ensure spring constants don't go negative
                if nn.w[l-1][i, j] < 0.01
                    nn.w[l-1][i, j] = 0.01
                end
            end
        end
    end
end

"""
    train!(nn::SpringMassNN, inputs::Vector{Vector{Float64}}, 
           targets::Vector{Vector{Float64}}, epochs::Int=200,
           steps::Int=100, dt::Float64=0.01, learning_rate::Float64=0.03)

Train the network on the given dataset.
"""
function train!(nn::SpringMassNN, inputs::Vector{Vector{Float64}}, 
                targets::Vector{Vector{Float64}}; 
                epochs::Int=200, steps::Int=100, dt::Float64=0.01, learning_rate::Float64=0.03)
    
    if length(inputs) != length(targets)
        throw(ArgumentError("Inputs and targets must have the same length"))
    end
    
    error_history = Float64[]
    energy_history = Float64[]
    
    println("Starting training...")
    
    for epoch in 1:epochs
        total_error = 0.0
        
        # Loop through training examples
        for example in 1:length(inputs)
            # Set input
            set_input!(nn, inputs[example])
            
            # Reset hidden and output layer dynamics
            for l in 2:nn.layers
                for i in 1:nn.neurons[l]
                    nn.v[l][i] = 0.0
                    nn.a[l][i] = 0.0
                end
            end
            
            # Let system settle
            for _ in 1:steps
                integration_step!(nn, dt)
            end
            
            # Calculate error
            error = sum((nn.x[end][i] - targets[example][i])^2 for i in 1:nn.neurons[end])
            total_error += error
            
            # Record energy
            push!(energy_history, network_energy(nn))
            
            # Set target for learning
            set_target!(nn, targets[example])
            
            # Update spring constants
            learn_step!(nn, learning_rate)
        end
        
        # Record training error
        push!(error_history, total_error)
        
        # Print progress
        if epoch % 10 == 0
            println("Epoch $epoch: Error = $total_error")
        end
    end
    
    return error_history, energy_history
end

"""
    test_network(nn::SpringMassNN, inputs::Vector{Vector{Float64}}, 
                targets::Vector{Vector{Float64}})

Test the network on the given dataset and print results.
"""
function test_network(nn::SpringMassNN, inputs::Vector{Vector{Float64}}, 
                     targets::Vector{Vector{Float64}})
    println("\nTesting Network Results:")
    println("-----------------------")
    
    for i in 1:length(inputs)
        prediction = predict(nn, inputs[i])
        @printf("Input: %s → Output: [%.3f] (Expected: %s)\n", 
                inputs[i], prediction[1], targets[i])
    end
end

"""
    visualize_network(nn::SpringMassNN)

Create a visualization of the network structure.
"""
function visualize_network(nn::SpringMassNN)
    p = plot(size=(600, 400), legend=false, title="Spring-Mass Neural Network")
    
    # Calculate layer positions
    layer_x = collect(1:nn.layers)
    
    # Plot nodes
    for l in 1:nn.layers
        for i in 1:nn.neurons[l]
            # Y position (spreads nodes vertically)
            y_pos = (i - (nn.neurons[l] + 1)/2) * 2
            
            # Node color by layer
            node_color = l == 1 ? :red : (l == nn.layers ? :blue : :green)
            
            # Plot node
            scatter!(p, [layer_x[l]], [y_pos], markersize=10, color=node_color)
            
            # Plot connections to previous layer
            if l > 1
                for j in 1:nn.neurons[l-1]
                    prev_y = (j - (nn.neurons[l-1] + 1)/2) * 2
                    
                    # Line width based on spring constant
                    w_val = nn.w[l-1][i, j]
                    line_width = abs(w_val) * 3
                    
                    # Plot connection
                    plot!(p, [layer_x[l-1], layer_x[l]], [prev_y, y_pos], 
                          linewidth=line_width, color=:gray, alpha=0.6)
                end
            end
        end
    end
    
    # Labels
    xlabel!(p, "Layer")
    ylabel!(p, "Node Position")
    
    return p
end

# Example: XOR problem solution
function run_xor_example()
    # Create network with 5 hidden units
    nn = SpringMassNN(3, [2, 5, 1])
    
    # XOR training data
    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    targets = [[0.0], [1.0], [1.0], [0.0]]
    
    # Train the network
    error_history, energy_history = train!(nn, inputs, targets, 
                                          epochs=200, steps=100, dt=0.01, learning_rate=0.03)
    
    # Test the network
    test_network(nn, inputs, targets)
    
    # Visualize results
    p1 = plot(error_history, title="Training Error", xlabel="Epoch", ylabel="Error", 
             legend=false, linewidth=2, color=:red)
    
    p2 = plot(energy_history, title="System Energy", xlabel="Example", ylabel="Energy", 
             legend=false, linewidth=2, color=:blue)
    
    p3 = visualize_network(nn)
    
    # Combine plots
    final_plot = plot(p1, p2, p3, layout=(3,1), size=(800, 900))
    display(final_plot)
    
    return nn, final_plot
end

# Display equations and run example
println("""
Spring-Mass Neural Network - Equations:
----------------------------------------
1. Energy Components:
   - Kinetic Energy: E_kin(i) = (1/2)m_i v_i^2
   - Sink Potential: E_sink(i) = (1/2)K_i x_i^2
   - Coupling Potential: E_coup(i,j) = (1/2)k_{i,j}(x_i - x_j)^2

2. Force Equation:
   F_i = -∇E_i = -K_i x_i - Σ k_{i,j}(x_i - x_j) - c_i v_i

3. Acceleration:
   a_i = F_i / m_i

4. Velocity Verlet Integration:
   v_i(t+dt/2) = v_i(t) + (dt/2)a_i(t)
   x_i(t+dt) = x_i(t) + dt*v_i(t+dt/2)
   a_i(t+dt) = F_i(x(t+dt)) / m_i
   v_i(t+dt) = v_i(t+dt/2) + (dt/2)a_i(t+dt)

5. Learning Rule:
   Δk = -η * ∂E/∂k = -η * (1/2)(x_i - x_j)^2
   k_new = k_old - Δk
""")

# Run the example
nn, plots = run_xor_example()