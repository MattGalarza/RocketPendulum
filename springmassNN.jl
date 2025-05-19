using Plots
using Random

"""
    SpringMassNeuralNetwork

A Neural Network based on spring-mass physical system dynamics.
"""
mutable struct SpringMassNeuralNetwork
    layer_sizes::Vector{Int}
    num_layers::Int
    weights::Vector{Matrix{Float64}}
    masses::Vector{Vector{Float64}}
    spring_constants::Vector{Matrix{Float64}}
    damping::Float64
    dt::Float64
    learning_rate::Float64
    
    function SpringMassNeuralNetwork(layer_sizes::Vector{Int}; 
                                    damping::Float64=0.2, 
                                    dt::Float64=0.01, 
                                    learning_rate::Float64=0.05)
        num_layers = length(layer_sizes)
        
        # Initialize weights randomly
        weights = Vector{Matrix{Float64}}(undef, num_layers-1)
        for i in 1:(num_layers-1)
            weights[i] = rand(Float64, layer_sizes[i+1], layer_sizes[i]) .- 0.5
        end
        
        # Initialize physical parameters
        # For simplicity, all masses are set to 1
        masses = [ones(Float64, size) for size in layer_sizes]
        
        # Spring constants could be adjusted, but set to 1 for now
        spring_constants = Vector{Matrix{Float64}}(undef, num_layers-1)
        for i in 1:(num_layers-1)
            spring_constants[i] = ones(Float64, layer_sizes[i+1], layer_sizes[i])
        end
        
        new(layer_sizes, num_layers, weights, masses, spring_constants, damping, dt, learning_rate)
    end
end

"""
    activation(x)

Activation function (tanh).
"""
function activation(x)
    return tanh.(x)
end

"""
    activation_derivative(x)

Derivative of the activation function.
"""
function activation_derivative(x)
    return 1.0 .- tanh.(x).^2
end

"""
    forward_pass(network, activities)

Calculate predictions and errors for all layers.
"""
function forward_pass(network, activities)
    predictions = [zeros(Float64, size(a)) for a in activities]
    errors = [zeros(Float64, size(a)) for a in activities]
    
    # Calculate predictions for each layer
    for l in 2:network.num_layers
        # Apply activation to previous layer activities
        prev_activations = activation(activities[l-1])
        
        # Compute weighted sum
        predictions[l] = network.weights[l-1] * prev_activations
        
        # Calculate error
        errors[l] = activities[l] - predictions[l]
    end
    
    return predictions, errors
end

"""
    calculate_energy(network, activities, predictions, velocities)

Calculate the system's total energy.
"""
function calculate_energy(network, activities, predictions, velocities)
    energy = 0.0
    
    # Kinetic energy
    for l in 1:network.num_layers
        energy += 0.5 * sum(network.masses[l] .* velocities[l].^2)
    end
    
    # Potential energy from prediction errors
    for l in 2:network.num_layers
        energy += 0.5 * sum((activities[l] - predictions[l]).^2)
    end
    
    return energy
end

"""
    update_state!(activities, velocities, network, predictions, errors)

Update neuron activities and velocities based on physics.
"""
function update_state!(activities, velocities, network, predictions, errors)
    # Update hidden layers (not input or output during training)
    for l in 2:(network.num_layers-1)
        # Calculate forces on neurons
        force = -errors[l]  # Error correction force
        
        # Force from next layer errors
        if l < network.num_layers
            prev_activations = activation(activities[l])
            next_layer_influence = transpose(network.weights[l]) * errors[l+1]
            force += next_layer_influence
        end
        
        # Add damping force
        force -= network.damping * velocities[l]
        
        # Apply F = ma to get acceleration (assuming mass = 1)
        acceleration = force ./ network.masses[l]
        
        # Update velocity
        velocities[l] += acceleration * network.dt
        
        # Update position/activity
        activities[l] += velocities[l] * network.dt
    end
end

"""
    update_weights!(network, activities, errors)

Update weights based on errors and activations.
"""
function update_weights!(network, activities, errors)
    for l in 1:(network.num_layers-1)
        # Hebbian-like learning rule
        prev_activations = activation(activities[l])
        
        # Weight change proportional to error and activation
        delta_w = network.learning_rate * errors[l+1] * transpose(prev_activations)
        
        # Add a small oscillatory component to help escape local minima
        # This is like giving the springs a small "shake" to find better configurations
        oscillation = 0.05 * network.learning_rate * (rand(size(delta_w)...) .- 0.5)
        
        network.weights[l] += delta_w .+ oscillation
    end
end

"""
    train!(network, inputs, targets; num_epochs=50, max_time=5.0, convergence_threshold=0.001)

Train the network.
"""
function train!(network, inputs, targets; 
                num_epochs=50, max_time=5.0, convergence_threshold=0.001)
    energy_history = Float64[]
    error_history = Float64[]
    
    println("Training the Spring-Mass Neural Network...")
    
    for epoch in 1:num_epochs
        total_error = 0.0
        last_energy = 0.0  # Track the last energy value
        
        # Process each training example
        for i in 1:size(inputs, 1)
            # Initialize activities and velocities
            activities = [zeros(Float64, size) for size in network.layer_sizes]
            velocities = [zeros(Float64, size) for size in network.layer_sizes]
            
            # Set input
            activities[1] = inputs[i, :]
            
            # Set target output
            activities[end] = targets[i, :]
            
            # Let the system settle
            for t in 0:network.dt:max_time
                # Forward pass
                predictions, errors = forward_pass(network, activities)
                
                # Update states
                update_state!(activities, velocities, network, predictions, errors)
                
                # Update weights
                update_weights!(network, activities, errors)
                
                # Calculate energy
                current_energy = calculate_energy(network, activities, predictions, velocities)
                last_energy = current_energy  # Store for later use
                
                # Stop if energy is low enough
                if current_energy < convergence_threshold
                    break
                end
            end
            
            # Calculate error for this example
            predictions, errors = forward_pass(network, activities)
            example_error = sum(errors[end].^2)
            total_error += example_error
        end
        
        # Record history
        push!(energy_history, last_energy)  # Use the stored energy value
        push!(error_history, total_error)
        
        # Print progress
        if epoch % 10 == 0
            println("Epoch $epoch: Error = $total_error")
        end
    end
    
    println("Training complete!")
    return energy_history, error_history
end

"""
    predict(network, input_data; max_time=5.0, convergence_threshold=0.001)

Make a prediction for a given input.
"""
function predict(network, input_data; max_time=10.0, convergence_threshold=0.0005)
    # Initialize activities and velocities
    activities = [zeros(Float64, size) for size in network.layer_sizes]
    velocities = [zeros(Float64, size) for size in network.layer_sizes]
    
    # Set input
    activities[1] = input_data
    
    # Let the system settle to find the output
    for t in 0:network.dt:max_time
        # Forward pass
        predictions, errors = forward_pass(network, activities)
        
        # Calculate forces and update all layers except input
        for l in 2:network.num_layers
            # Calculate force
            force = -errors[l]
            
            # Add damping
            force -= network.damping * velocities[l]
            
            # Update velocity and position
            velocities[l] += force * network.dt
            activities[l] += velocities[l] * network.dt
        end
        
        # Check convergence
        energy = sum(sum(errors[l].^2) for l in 2:network.num_layers)
        if energy < convergence_threshold
            break
        end
    end
    
    # Apply a sigmoid-like scaling to the output for better 0/1 separation
    # This helps transform the small output values to more clearly separated results
    outputs = activities[end]
    scaled_outputs = 1.0 ./ (1.0 .+ exp.(-10.0 * outputs))
    
    return scaled_outputs
end

"""
    test_xor()

Test the Spring-Mass Neural Network on the XOR problem.
"""
function test_xor()
    # XOR problem data
    inputs = [0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0]
    targets = [0.0; 1.0; 1.0; 0.0]
    
    # Reshape targets to be a column vector for each example
    targets = reshape(targets, (4, 1))
    
    # Create network with 2 inputs, 4 hidden neurons, 1 output
    # Use physics parameters specifically tuned for XOR
    network = SpringMassNeuralNetwork([2, 4, 1], 
                                     damping=0.1,  # Lower damping allows more oscillation
                                     dt=0.02,      # Larger time step for faster dynamics
                                     learning_rate=0.2) # Higher learning rate for stronger "springs"
    
    # Initialize weights with slightly larger values
    for l in 1:length(network.weights)
        network.weights[l] = 2.0 * (rand(Float64, size(network.weights[l])...) .- 0.5)
    end
    
    # Train the network with more time per example to allow settling
    energy_history, error_history = train!(network, inputs, targets, 
                                          num_epochs=100, 
                                          max_time=10.0, 
                                          convergence_threshold=0.0005)
    
    # Plot training progress
    p = plot(error_history, 
            title="Training Error Over Time for XOR", 
            xlabel="Epoch", 
            ylabel="Error", 
            grid=true, 
            linewidth=2,
            legend=false)
    display(p)
    
    # Test the network
    println("\nTesting Spring-Mass Neural Network on XOR:")
    println("--------------------------------")
    for i in 1:size(inputs, 1)
        prediction = predict(network, inputs[i, :])
        println("Input: $(inputs[i, :]) → Output: $(round(prediction[1], digits=4)) (Expected: $(targets[i, 1]))")
    end
end

"""
    generate_sparse_sinusoidal_dataset(num_samples, input_dim, num_components)

Generate a dataset of sparse sinusoidal signals and their transformations.
Each signal is a sum of a small number of sinusoids with random frequencies and amplitudes.
The target output is a specific transformation of this signal.
"""
function generate_sparse_sinusoidal_dataset(num_samples, input_dim, num_components; seed=42)
    Random.seed!(seed)  # For reproducibility
    
    # Frequency range
    min_freq = 0.5
    max_freq = 5.0
    
    # Generate dataset
    inputs = zeros(num_samples, input_dim)
    features = zeros(num_samples, num_components * 3)  # Store frequency, amplitude, phase for each component
    targets = zeros(num_samples, 1)
    
    time_points = range(0, 1, length=input_dim)
    
    for i in 1:num_samples
        # Generate random frequencies, amplitudes, and phases for each component
        frequencies = min_freq .+ (max_freq - min_freq) * rand(num_components)
        amplitudes = 0.3 .+ 0.7 * rand(num_components)
        phases = 2π * rand(num_components)
        
        # Store the signal parameters
        for j in 1:num_components
            features[i, (j-1)*3+1] = frequencies[j]
            features[i, (j-1)*3+2] = amplitudes[j]
            features[i, (j-1)*3+3] = phases[j]
        end
        
        # Generate the sparse sinusoidal signal
        signal = zeros(input_dim)
        for j in 1:num_components
            signal .+= amplitudes[j] * sin.(2π * frequencies[j] * time_points .+ phases[j])
        end
        
        # Normalize to range [-1, 1]
        max_val = maximum(abs.(signal))
        if max_val > 0
            signal ./= max_val
        end
        
        inputs[i, :] = signal
        
        # Target: A non-linear function of the dominant frequency and amplitude
        # Here we use the weighted sum of frequencies, where weights are the squared amplitudes
        target = sum(frequencies .* (amplitudes.^2)) / sum(amplitudes.^2)
        targets[i, 1] = target
    end
    
    return inputs, targets, features
end

"""
    visualize_dataset(inputs, targets, features; num_examples=5)

Visualize a few examples from the sparse sinusoidal dataset.
"""
function visualize_dataset(inputs, targets, features; num_examples=5)
    num_examples = min(num_examples, size(inputs, 1))
    n_rows = num_examples
    
    p = plot(layout=(n_rows, 1), size=(800, 200*n_rows), legend=false)
    
    for i in 1:num_examples
        time_points = range(0, 1, length=size(inputs, 2))
        plot!(p, time_points, inputs[i, :], subplot=i, 
              title="Example $i: Target = $(round(targets[i, 1], digits=3))", 
              ylabel="Amplitude", 
              xlabel=(i == num_examples ? "Time" : ""))
    end
    
    return p
end

"""
    test_spring_mass_nn_on_sinusoidal_data()

Test the Spring-Mass Neural Network on a sparse sinusoidal mapping task.
"""
function test_spring_mass_nn_on_sinusoidal_data()
    # Dataset parameters
    num_samples = 100
    input_dim = 50
    num_components = 3
    
    # Generate dataset
    println("Generating sparse sinusoidal dataset...")
    inputs, targets, features = generate_sparse_sinusoidal_dataset(num_samples, input_dim, num_components)
    
    # Visualize some examples
    p_examples = visualize_dataset(inputs, targets, features, num_examples=3)
    display(p_examples)
    
    # Split into training and testing sets
    train_ratio = 0.8
    num_train = Int(floor(num_samples * train_ratio))
    
    train_inputs = inputs[1:num_train, :]
    train_targets = targets[1:num_train, :]
    test_inputs = inputs[num_train+1:end, :]
    test_targets = targets[num_train+1:end, :]
    
    println("Dataset created with $(num_train) training and $(num_samples - num_train) testing examples")
    
    # Create network with appropriate architecture for this problem
    # Input layer: input_dim neurons
    # Hidden layers: Two hidden layers with decreasing size
    # Output layer: 1 neuron
    hidden_size1 = min(100, input_dim * 2)
    hidden_size2 = min(50, hidden_size1 ÷ 2)
    
    println("Creating Spring-Mass Neural Network with architecture: [$input_dim, $hidden_size1, $hidden_size2, 1]")
    network = SpringMassNeuralNetwork([input_dim, hidden_size1, hidden_size2, 1], 
                                      damping=0.3, dt=0.005, learning_rate=0.01)
    
    # Train the network
    num_epochs = 50
    println("Training the network for $num_epochs epochs...")
    energy_history, error_history = train!(network, train_inputs, train_targets, 
                                          num_epochs=num_epochs, max_time=3.0)
    
    # Plot training progress
    p_training = plot(error_history, 
                    title="Training Error Over Time", 
                    xlabel="Epoch", 
                    ylabel="Error", 
                    grid=true, 
                    linewidth=2,
                    legend=false)
    display(p_training)
    
    # Evaluate on test set
    println("\nEvaluating on test set...")
    test_predictions = zeros(size(test_targets))
    for i in 1:size(test_inputs, 1)
        test_predictions[i, :] = predict(network, test_inputs[i, :])
    end
    
    # Calculate MSE
    test_mse = mean((test_predictions .- test_targets).^2)
    println("Test MSE: $test_mse")
    
    # Visualize predictions vs targets
    p_results = scatter(test_targets, test_predictions, 
                      xlabel="True Values", 
                      ylabel="Predicted Values", 
                      title="Spring-Mass NN Predictions on Sinusoidal Mapping", 
                      legend=false)
    
    # Add diagonal line representing perfect predictions
    min_val = min(minimum(test_targets), minimum(test_predictions))
    max_val = max(maximum(test_targets), maximum(test_predictions))
    range_vals = range(min_val, max_val, length=100)
    plot!(p_results, range_vals, range_vals, linestyle=:dash, color=:red)
    
    display(p_results)
    
    # Visualize a few examples with their predictions
    println("\nVisualizing examples with predictions...")
    num_viz = 3
    p_viz = plot(layout=(num_viz, 1), size=(800, 200*num_viz))
    
    for i in 1:num_viz
        idx = i
        time_points = range(0, 1, length=size(test_inputs, 2))
        plot!(p_viz, time_points, test_inputs[idx, :], subplot=i, 
              title="Example $i: True = $(round(test_targets[idx, 1], digits=3)), Predicted = $(round(test_predictions[idx, 1], digits=3))", 
              ylabel="Amplitude", 
              xlabel=(i == num_viz ? "Time" : ""))
    end
    
    display(p_viz)
    
    return network, p_results
end

# Main function to run all tests
function main()
    println("=== Testing Spring-Mass Neural Network on XOR Problem ===")
    test_xor()
    
    println("\n\n=== Testing Spring-Mass Neural Network on Sinusoidal Mapping ===")
    test_spring_mass_nn_on_sinusoidal_data()
end

# Run the main function
main()