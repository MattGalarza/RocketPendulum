using Plots
using Random

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

function main()
    # Run the sinusoidal signal mapping test
    test_spring_mass_nn_on_sinusoidal_data()
end

# Run the main function if this is the main script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
network = SpringMassNeuralNetwork([2, 3, 1])
energy_history, error_history = train!(network, inputs, targets)
output = predict(network, input_data)