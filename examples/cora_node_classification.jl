using Random
using Statistics
using LinearAlgebra
using Lux
using Optimisers
using NNlib
using SimpleHypergraphs
using HyperGraphNeuralNetworks
using Enzyme

const RANDOM_SEED = 1234
const HIDDEN_DIM = 64
const TRAIN_FRACTION = 0.60
const VALIDATION_FRACTION = 0.20
const NUMBER_OF_EPOCHS = 100
const LEARNING_RATE = 1.0f-3

println()
println("Cora Node Classification Proof of Concept")
println("Random seed: ", RANDOM_SEED)

Random.seed!(RANDOM_SEED)

function load_cora_dataset(
    content_path::AbstractString,
    citations_path::AbstractString,
)
    println("Loading Cora dataset...")

    isfile(content_path) ||
        error("Cora content file not found: $content_path")

    isfile(citations_path) ||
        error("Cora citation file not found: $citations_path")

    content_lines = readlines(content_path)

    isempty(content_lines) &&
        error("The Cora content file is empty.")

    first_fields = split(strip(content_lines[1]))

    number_of_features = length(first_fields) - 2

    number_of_features > 0 ||
        error("Could not determine the Cora feature dimension.")

    number_of_papers = length(content_lines)

    paper_ids = Vector{Int}(undef, number_of_papers)

    X = zeros(
        Float32,
        number_of_papers,
        number_of_features,
    )

    raw_labels = Vector{String}(undef, number_of_papers)

    for (row_index, line) in enumerate(content_lines)
        fields = split(strip(line))

        expected_fields = number_of_features + 2

        length(fields) == expected_fields ||
            error(
                "Malformed Cora content row $row_index. " *
                "Expected $expected_fields fields but found $(length(fields))."
            )

        paper_ids[row_index] = parse(Int, fields[1])

        for feature_index in 1:number_of_features
            X[row_index, feature_index] =
                parse(Float32, fields[feature_index + 1])
        end

        raw_labels[row_index] = fields[end]
    end

    class_names = sort(unique(raw_labels))

    class_to_index = Dict(
        class_name => index
        for (index, class_name) in enumerate(class_names)
    )

    y = [
        class_to_index[label]
        for label in raw_labels
    ]

    citation_lines = readlines(citations_path)

    citations = Tuple{Int, Int}[]

    for (line_number, line) in enumerate(citation_lines)
        fields = split(strip(line))

        length(fields) == 2 ||
            error(
                "Malformed Cora citation row $line_number."
            )

        source_id = parse(Int, fields[1])
        target_id = parse(Int, fields[2])

        push!(
            citations,
            (source_id, target_id),
        )
    end

    println()
    println("Dataset summary")
    println("Number of papers: ", number_of_papers)
    println("Number of input features: ", number_of_features)
    println("Number of citation records: ", length(citations))
    println("Number of classes: ", length(class_names))
    println("Class mapping: ", class_to_index)

    println()
    println("Class distribution:")

    for class_name in class_names
        class_index = class_to_index[class_name]

        count_in_class = count(
            ==(class_index),
            y,
        )

        percentage =
            100 * count_in_class / number_of_papers

        println(
            "  ",
            class_name,
            ": ",
            count_in_class,
            " (",
            round(percentage; digits = 2),
            "%)",
        )
    end

    return (
        paper_ids = paper_ids,
        X = X,
        y = Int.(y),
        raw_labels = raw_labels,
        class_names = class_names,
        class_to_index = class_to_index,
        citations = citations,
        number_of_papers = number_of_papers,
        number_of_features = number_of_features,
        number_of_classes = length(class_names),
    )
end

function build_paper_id_mapping(paper_ids)
    id_to_index = Dict{Int, Int}()

    for (index, paper_id) in enumerate(paper_ids)
        haskey(id_to_index, paper_id) &&
            error("Duplicate paper ID found: $paper_id")

        id_to_index[paper_id] = index
    end

    println()
    println(
        "Paper ID mapping created for ",
        length(id_to_index),
        " papers.",
    )

    return id_to_index
end

function validate_citations(
    citations,
    id_to_index,
)
    valid_citations = Tuple{Int, Int}[]
    discarded_citations = 0

    for (source_id, target_id) in citations
        if haskey(id_to_index, source_id) &&
           haskey(id_to_index, target_id)

            push!(
                valid_citations,
                (source_id, target_id),
            )
        else
            discarded_citations += 1
        end
    end

    println()
    println("Citation validation")
    println("Original citations: ", length(citations))
    println("Valid citations: ", length(valid_citations))
    println("Discarded citations: ", discarded_citations)

    isempty(valid_citations) &&
        error("No valid citations remain after validation.")

    return valid_citations
end

function build_directed_incidence_matrices(
    valid_citations,
    id_to_index,
    number_of_vertices,
)
    number_of_hyperedges = length(valid_citations)

    incidence_tail = zeros(
        Float32,
        number_of_vertices,
        number_of_hyperedges,
    )

    incidence_head = zeros(
        Float32,
        number_of_vertices,
        number_of_hyperedges,
    )

    for (
        hyperedge_index,
        (source_id, target_id),
    ) in enumerate(valid_citations)

        source_index = id_to_index[source_id]
        target_index = id_to_index[target_id]

        incidence_tail[
            source_index,
            hyperedge_index,
        ] = 1.0f0

        incidence_head[
            target_index,
            hyperedge_index,
        ] = 1.0f0
    end

    println()
    println("Directed incidence representation")
    println(
        "Tail incidence matrix size: ",
        size(incidence_tail),
    )
    println(
        "Head incidence matrix size: ",
        size(incidence_head),
    )
    println(
        "Tail memberships: ",
        Int(sum(incidence_tail)),
    )
    println(
        "Head memberships: ",
        Int(sum(incidence_head)),
    )

    size(incidence_tail) == size(incidence_head) ||
        error("Tail and head incidence matrices have different sizes.")

    all(
        vec(sum(incidence_tail; dims = 1)) .== 1.0f0
    ) || error(
        "Each citation hyperedge must have exactly one tail vertex."
    )

    all(
        vec(sum(incidence_head; dims = 1)) .== 1.0f0
    ) || error(
        "Each citation hyperedge must have exactly one head vertex."
    )

    println("Incidence matrix validation passed.")

    return incidence_tail, incidence_head
end
function construct_hgnn_dihypergraph(
    incidence_tail,
    incidence_head,
)
    number_of_vertices,
    number_of_hyperedges = size(incidence_tail)

    println()
    println("Constructing HGNNDiHypergraph...")

    tail_hypergraph = Hypergraph{Float32}(
        number_of_vertices,
        number_of_hyperedges,
    )

    head_hypergraph = Hypergraph{Float32}(
        number_of_vertices,
        number_of_hyperedges,
    )

    for hyperedge_index in 1:number_of_hyperedges
        tail_vertices = findall(
            !iszero,
            view(
                incidence_tail,
                :,
                hyperedge_index,
            ),
        )

        head_vertices = findall(
            !iszero,
            view(
                incidence_head,
                :,
                hyperedge_index,
            ),
        )

        for vertex_index in tail_vertices
            tail_hypergraph[
                vertex_index,
                hyperedge_index,
            ] = incidence_tail[
                vertex_index,
                hyperedge_index,
            ]
        end

        for vertex_index in head_vertices
            head_hypergraph[
                vertex_index,
                hyperedge_index,
            ] = incidence_head[
                vertex_index,
                hyperedge_index,
            ]
        end
    end

    directed_hypergraph =
        HGNNDiHypergraph{
            Float32,
            Dict{Int, Float32},
        }(
            tail_hypergraph,
            head_hypergraph,
        )

    println(
        "Directed hypergraph successfully created."
    )

    println(
        "Hypergraph vertices: ",
        number_of_vertices,
    )

    println(
        "Hypergraph hyperedges: ",
        number_of_hyperedges,
    )

    println(
        "HGNNDiHypergraph validation passed."
    )

    return directed_hypergraph
end
function compute_directed_graph_statistics(
    incidence_tail,
    incidence_head,
)
    outgoing_degree =
        vec(sum(incidence_tail; dims = 2))

    incoming_degree =
        vec(sum(incidence_head; dims = 2))

    total_degree =
        outgoing_degree .+ incoming_degree

    isolated_vertices =
        count(iszero, total_degree)

    statistics = (
        mean_outgoing_degree = mean(outgoing_degree),
        mean_incoming_degree = mean(incoming_degree),
        mean_total_degree = mean(total_degree),
        maximum_outgoing_degree = maximum(outgoing_degree),
        maximum_incoming_degree = maximum(incoming_degree),
        maximum_total_degree = maximum(total_degree),
        isolated_vertices = isolated_vertices,
    )

    println()
    println("Directed graph statistics")

    println(
        "Mean outgoing degree: ",
        round(
            statistics.mean_outgoing_degree;
            digits = 3,
        ),
    )

    println(
        "Mean incoming degree: ",
        round(
            statistics.mean_incoming_degree;
            digits = 3,
        ),
    )

    println(
        "Mean total degree: ",
        round(
            statistics.mean_total_degree;
            digits = 3,
        ),
    )

    println(
        "Maximum outgoing degree: ",
        statistics.maximum_outgoing_degree,
    )

    println(
        "Maximum incoming degree: ",
        statistics.maximum_incoming_degree,
    )

    println(
        "Maximum total degree: ",
        statistics.maximum_total_degree,
    )

    println(
        "Isolated vertices: ",
        statistics.isolated_vertices,
    )

    return statistics
end

function row_normalise_features(X)
    row_sums = sum(X; dims = 2)

    safe_row_sums = ifelse.(
        iszero.(row_sums),
        one(eltype(row_sums)),
        row_sums,
    )

    return X ./ safe_row_sums
end

function preprocess_features(X)
    println()
    println("Feature preprocessing")

    X_normalised = row_normalise_features(X)

    println(
        "Original feature matrix size: ",
        size(X),
    )

    println(
        "Normalised feature matrix size: ",
        size(X_normalised),
    )

    println(
        "Non-zero original features: ",
        count(!iszero, X),
    )

    println(
        "Non-zero normalised features: ",
        count(!iszero, X_normalised),
    )

    zero_feature_vertices = count(
        iszero,
        vec(sum(abs.(X); dims = 2)),
    )

    println(
        "Vertices with no active features: ",
        zero_feature_vertices,
    )

    all(isfinite, X_normalised) ||
        error(
            "Feature preprocessing produced non-finite values."
        )

    println("Feature preprocessing complete.")

    return Float32.(X_normalised)
end

function stratified_split(
    y;
    train_fraction = TRAIN_FRACTION,
    validation_fraction = VALIDATION_FRACTION,
    seed = RANDOM_SEED,
)
    train_fraction > 0 ||
        throw(
            ArgumentError(
                "`train_fraction` must be positive."
            )
        )

    validation_fraction > 0 ||
        throw(
            ArgumentError(
                "`validation_fraction` must be positive."
            )
        )

    train_fraction + validation_fraction < 1 ||
        throw(
            ArgumentError(
                "Training and validation fractions must sum to less than one."
            )
        )

    rng = MersenneTwister(seed)

    train_indices = Int[]
    validation_indices = Int[]
    test_indices = Int[]

    classes = sort(unique(y))

    for class_label in classes
        class_indices = findall(
            ==(class_label),
            y,
        )

        shuffle!(rng, class_indices)

        number_in_class = length(class_indices)

        number_train = floor(
            Int,
            train_fraction * number_in_class,
        )

        number_validation = floor(
            Int,
            validation_fraction * number_in_class,
        )

        train_end = number_train
        validation_start = train_end + 1
        validation_end =
            train_end + number_validation
        test_start = validation_end + 1

        if train_end >= 1
            append!(
                train_indices,
                class_indices[1:train_end],
            )
        end

        if validation_start <= validation_end
            append!(
                validation_indices,
                class_indices[
                    validation_start:validation_end
                ],
            )
        end

        if test_start <= number_in_class
            append!(
                test_indices,
                class_indices[
                    test_start:number_in_class
                ],
            )
        end
    end

    shuffle!(rng, train_indices)
    shuffle!(rng, validation_indices)
    shuffle!(rng, test_indices)

    return (
        train = train_indices,
        validation = validation_indices,
        test = test_indices,
    )
end

function print_split_distribution(
    split_name,
    indices,
    y,
    class_names,
)
    println()
    println(split_name, " split:")
    println("  Total nodes: ", length(indices))

    split_labels = y[indices]

    for (
        class_index,
        class_name,
    ) in enumerate(class_names)

        class_count = count(
            ==(class_index),
            split_labels,
        )

        percentage =
            100 * class_count / length(indices)

        println(
            "  ",
            class_name,
            ": ",
            class_count,
            " (",
            round(percentage; digits = 2),
            "%)",
        )
    end
end

function prepare_cora()
    content_path = joinpath(
        @__DIR__,
        "..",
        "data",
        "cora",
        "cora.content",
    )

    citations_path = joinpath(
        @__DIR__,
        "..",
        "data",
        "cora",
        "cora.cites",
    )

    dataset = load_cora_dataset(
        content_path,
        citations_path,
    )

    id_to_index =
        build_paper_id_mapping(
            dataset.paper_ids,
        )

    valid_citations =
        validate_citations(
            dataset.citations,
            id_to_index,
        )

    incidence_tail,
    incidence_head =
        build_directed_incidence_matrices(
            valid_citations,
            id_to_index,
            dataset.number_of_papers,
        )

    hypergraph = nothing 
        

    graph_statistics =
        compute_directed_graph_statistics(
            incidence_tail,
            incidence_head,
        )

    X =
        preprocess_features(
            dataset.X,
        )

    split =
        stratified_split(
            dataset.y;
            train_fraction = TRAIN_FRACTION,
            validation_fraction = VALIDATION_FRACTION,
            seed = RANDOM_SEED,
        )

    train_indices = split.train
    validation_indices = split.validation
    test_indices = split.test

    println()
    println("Data split summary")
    println(
        "Training nodes: ",
        length(train_indices),
    )
    println(
        "Validation nodes: ",
        length(validation_indices),
    )
    println(
        "Test nodes: ",
        length(test_indices),
    )

    print_split_distribution(
        "Training",
        train_indices,
        dataset.y,
        dataset.class_names,
    )

    print_split_distribution(
        "Validation",
        validation_indices,
        dataset.y,
        dataset.class_names,
    )

    print_split_distribution(
        "Test",
        test_indices,
        dataset.y,
        dataset.class_names,
    )

    println()
    println("Running final consistency checks...")

    number_of_vertices =
        dataset.number_of_papers

    number_of_hyperedges =
        length(valid_citations)

    size(X, 1) == number_of_vertices ||
        error("Feature matrix vertex count mismatch.")

    length(dataset.y) == number_of_vertices ||
        error("Label count mismatch.")

    size(incidence_tail) ==
        (
            number_of_vertices,
            number_of_hyperedges,
        ) ||
        error("Tail incidence matrix size mismatch.")

    size(incidence_head) ==
        (
            number_of_vertices,
            number_of_hyperedges,
        ) ||
        error("Head incidence matrix size mismatch.")

    all_indices = vcat(
        train_indices,
        validation_indices,
        test_indices,
    )

    length(all_indices) == number_of_vertices ||
        error(
            "Train, validation and test splits do not cover every node."
        )

    length(unique(all_indices)) ==
        number_of_vertices ||
        error(
            "Train, validation and test splits overlap."
        )

    minimum(dataset.y) == 1 ||
        error("Class indices must start at one.")

    maximum(dataset.y) ==
        dataset.number_of_classes ||
        error("Class index mismatch.")

    println("All consistency checks passed.")

    println()
    println(
        "Cora proof-of-concept data preparation complete"
    )
    println(
        "Nodes: ",
        number_of_vertices,
    )
    println(
        "Directed hyperedges: ",
        number_of_hyperedges,
    )
    println(
        "Features per node: ",
        dataset.number_of_features,
    )
    println(
        "Prediction classes: ",
        dataset.number_of_classes,
    )
    println(
        "Training nodes: ",
        length(train_indices),
    )
    println(
        "Validation nodes: ",
        length(validation_indices),
    )
    println(
        "Test nodes: ",
        length(test_indices),
    )

    return (
        hypergraph = hypergraph,
        X = X,
        y = dataset.y,
        incidence_tail = incidence_tail,
        incidence_head = incidence_head,
        train_indices = train_indices,
        validation_indices = validation_indices,
        test_indices = test_indices,
        class_names = dataset.class_names,
        class_to_index = dataset.class_to_index,
        paper_ids = dataset.paper_ids,
        id_to_index = id_to_index,
        graph_statistics = graph_statistics,
        number_of_vertices = number_of_vertices,
        number_of_hyperedges = number_of_hyperedges,
        number_of_features = dataset.number_of_features,
        number_of_classes = dataset.number_of_classes,
        seed = RANDOM_SEED,
    )
end
struct CoraDirectedClassifier{L, IW, IB} <: Lux.AbstractLuxLayer
    directed_layer::L
    hidden_dim::Int
    number_of_classes::Int
    init_weight::IW
    init_bias::IB
end

function CoraDirectedClassifier(
    input_dim::Int,
    hidden_dim::Int,
    number_of_classes::Int;
    activation = tanh,
    normalize::Bool = true,
    init_weight = Lux.glorot_uniform,
    init_bias = Lux.zeros32,
)
    input_dim > 0 ||
        throw(
            ArgumentError(
                "`input_dim` must be positive."
            )
        )

    hidden_dim > 0 ||
        throw(
            ArgumentError(
                "`hidden_dim` must be positive."
            )
        )

    number_of_classes > 1 ||
        throw(
            ArgumentError(
                "`number_of_classes` must be greater than one."
            )
        )

    directed_layer = DirectedHypergraphLayer(
        input_dim,
        0,
        hidden_dim;
        activation = activation,
        normalize = normalize,
        init_weight = init_weight,
        init_bias = init_bias,
    )

    return CoraDirectedClassifier(
        directed_layer,
        hidden_dim,
        number_of_classes,
        init_weight,
        init_bias,
    )
end

function initialise_classifier_weight(
    initializer,
    rng::AbstractRNG,
    input_dimension::Int,
    output_dimension::Int,
)
    return permutedims(
        initializer(
            rng,
            output_dimension,
            input_dimension,
        )
    )
end

function initialise_classifier_bias(
    initializer,
    rng::AbstractRNG,
    output_dimension::Int,
)
    return permutedims(
        initializer(
            rng,
            output_dimension,
            1,
        )
    )
end

function Lux.initialparameters(
    rng::AbstractRNG,
    model::CoraDirectedClassifier,
)
    directed_parameters =
        Lux.initialparameters(
            rng,
            model.directed_layer,
        )

    W_output =
        initialise_classifier_weight(
            model.init_weight,
            rng,
            model.hidden_dim,
            model.number_of_classes,
        )

    b_output =
        initialise_classifier_bias(
            model.init_bias,
            rng,
            model.number_of_classes,
        )

    return (
        directed_layer = directed_parameters,
        W_output = W_output,
        b_output = b_output,
    )
end

function Lux.initialstates(
    rng::AbstractRNG,
    model::CoraDirectedClassifier,
)
    directed_state =
        Lux.initialstates(
            rng,
            model.directed_layer,
        )

    return (
        directed_layer = directed_state,
    )
end

function Lux.parameterlength(
    model::CoraDirectedClassifier,
)
    directed_parameters =
        Lux.parameterlength(
            model.directed_layer,
        )

    output_parameters =
        model.hidden_dim *
        model.number_of_classes +
        model.number_of_classes

    return directed_parameters +
           output_parameters
end

function Lux.statelength(
    model::CoraDirectedClassifier,
)
    return Lux.statelength(
        model.directed_layer,
    )
end

function cora_forward(
    model::CoraDirectedClassifier,
    input,
    ps,
    st,
)
    X,
    incidence_tail,
    incidence_head = input

    directed_output,
    new_directed_state =
        model.directed_layer(
            (
                X,
                incidence_tail,
                incidence_head,
            ),
            ps.directed_layer,
            st.directed_layer,
        )

    hidden_vertices =
        directed_output.updated_vertices

    logits =
        hidden_vertices *
        ps.W_output .+
        ps.b_output

    new_state = (
        directed_layer = new_directed_state,
    )

    output = (
        logits = logits,
        hidden_vertices = hidden_vertices,
        hidden_hyperedges =
            directed_output.updated_hyperedges,
    )

    return output, new_state
end

function log_softmax_rows(logits)
    maximum_logits =
        maximum(
            logits;
            dims = 2,
        )

    shifted_logits =
        logits .- maximum_logits

    log_denominator =
        log.(
            sum(
                exp.(shifted_logits);
                dims = 2,
            )
        )

    return shifted_logits .-
           log_denominator
end

function classification_loss(
    logits,
    labels,
    indices,
)
    isempty(indices) &&
        throw(
            ArgumentError(
                "Cannot calculate loss for an empty index set."
            )
        )

    selected_logits =
        logits[
            indices,
            :,
        ]

    selected_labels =
        labels[
            indices
        ]

    log_probabilities =
        log_softmax_rows(
            selected_logits,
        )

    total_loss =
        zero(eltype(logits))

    for (
        local_index,
        class_index,
    ) in enumerate(selected_labels)

        total_loss -=
            log_probabilities[
                local_index,
                class_index,
            ]
    end

    return total_loss /
           length(selected_labels)
end

function predicted_classes(
    logits,
)
    number_of_vertices =
        size(logits, 1)

    predictions =
        Vector{Int}(
            undef,
            number_of_vertices,
        )

    for vertex_index in 1:number_of_vertices
        predictions[vertex_index] =
            argmax(
                view(
                    logits,
                    vertex_index,
                    :,
                )
            )
    end

    return predictions
end

function classification_accuracy(
    logits,
    labels,
    indices,
)
    isempty(indices) &&
        return 0.0

    predictions =
        predicted_classes(
            logits,
        )

    correct =
        count(
            predictions[indices] .==
            labels[indices]
        )

    return correct /
           length(indices)
end

function confusion_matrix(
    logits,
    labels,
    indices,
    number_of_classes,
)
    predictions =
        predicted_classes(
            logits,
        )

    matrix =
        zeros(
            Int,
            number_of_classes,
            number_of_classes,
        )

    for vertex_index in indices
        actual =
            labels[vertex_index]

        predicted =
            predictions[vertex_index]

        matrix[
            actual,
            predicted,
        ] += 1
    end

    return matrix
end

function per_class_accuracy(
    logits,
    labels,
    indices,
    class_names,
)
    predictions =
        predicted_classes(
            logits,
        )

    results =
        NamedTuple[]

    for (
        class_index,
        class_name,
    ) in enumerate(class_names)

        class_indices =
            [
                index
                for index in indices
                if labels[index] ==
                   class_index
            ]

        number_in_class =
            length(class_indices)

        correct =
            number_in_class == 0 ?
            0 :
            count(
                predictions[class_indices] .==
                class_index
            )

        accuracy =
            number_in_class == 0 ?
            0.0 :
            correct / number_in_class

        push!(
            results,
            (
                class_name = class_name,
                total = number_in_class,
                correct = correct,
                accuracy = accuracy,
            ),
        )
    end

    return results
end

function print_per_class_accuracy(
    results,
)
    println()
    println("Per-class test accuracy")

    for result in results
        println(
            "  ",
            result.class_name,
            ": ",
            round(
                100 * result.accuracy;
                digits = 2,
            ),
            "% (",
            result.correct,
            "/",
            result.total,
            ")",
        )
    end
end

function print_confusion_matrix(
    matrix,
    class_names,
)
    println()
    println("Test confusion matrix")
    println(
        "Rows represent true classes; columns represent predicted classes."
    )

    print("      ")

    for class_index in eachindex(class_names)
        print(
            lpad(
                string(class_index),
                6,
            )
        )
    end

    println()

    for row_index in axes(matrix, 1)
        print(
            lpad(
                string(row_index),
                5,
            ),
            " ",
        )

        for column_index in axes(matrix, 2)
            print(
                lpad(
                    string(
                        matrix[
                            row_index,
                            column_index,
                        ]
                    ),
                    6,
                )
            )
        end

        println()
    end

    println()
    println("Class index key:")

    for (
        class_index,
        class_name,
    ) in enumerate(class_names)
        println(
            "  ",
            class_index,
            " = ",
            class_name,
        )
    end
end

function print_prediction_examples(
    logits,
    labels,
    indices,
    paper_ids,
    class_names;
    number_to_show = 15,
)
    predictions =
        predicted_classes(
            logits,
        )

    number_to_show =
        min(
            number_to_show,
            length(indices),
        )

    println()
    println("Example test predictions")

    for vertex_index in indices[
        1:number_to_show
    ]
        true_class =
            class_names[
                labels[vertex_index]
            ]

        predicted_class =
            class_names[
                predictions[vertex_index]
            ]

        correct =
            labels[vertex_index] ==
            predictions[vertex_index]

        println(
            "  Paper ",
            paper_ids[vertex_index],
            " | true = ",
            true_class,
            " | predicted = ",
            predicted_class,
            " | correct = ",
            correct,
        )
    end
end

function model_summary(
    model,
    ps,
    st,
    cora_data,
)
    println()
    println(
        "Cora Directed Hypergraph Neural Network"
    )

    println()
    println("Model configuration")

    println(
        "Input dimension: ",
        cora_data.number_of_features,
    )

    println(
        "Hidden dimension: ",
        model.hidden_dim,
    )

    println(
        "Output classes: ",
        model.number_of_classes,
    )

    println(
        "Directed hypergraph layer: DirectedHypergraphLayer"
    )

    println()
    println(
        "Model parameter count: ",
        Lux.parameterlength(model),
    )

    println(
        "Model state count: ",
        Lux.statelength(model),
    )

    println()
    println("Parameter shapes")

    println(
        "  Directed W_vertex: ",
        size(
            ps.directed_layer.W_vertex
        ),
    )

    println(
        "  Directed b_vertex: ",
        size(
            ps.directed_layer.b_vertex
        ),
    )

    println(
        "  Directed W_hyperedge: ",
        size(
            ps.directed_layer.W_hyperedge
        ),
    )

    println(
        "  Directed b_hyperedge: ",
        size(
            ps.directed_layer.b_hyperedge
        ),
    )

    println(
        "  Directed W_vertex_update: ",
        size(
            ps.directed_layer.W_vertex_update
        ),
    )

    println(
        "  Directed b_vertex_update: ",
        size(
            ps.directed_layer.b_vertex_update
        ),
    )

    println(
        "  Output W_output: ",
        size(ps.W_output),
    )

    println(
        "  Output b_output: ",
        size(ps.b_output),
    )

    println(
        "  State keys: ",
        keys(st),
    )
end

function evaluate_model(
    model,
    ps,
    st,
    cora_data,
)
    input = (
        cora_data.X,
        cora_data.incidence_tail,
        cora_data.incidence_head,
    )

    output,
    new_state =
        cora_forward(
            model,
            input,
            ps,
            st,
        )

    train_loss =
        classification_loss(
            output.logits,
            cora_data.y,
            cora_data.train_indices,
        )

    validation_loss =
        classification_loss(
            output.logits,
            cora_data.y,
            cora_data.validation_indices,
        )

    test_loss =
        classification_loss(
            output.logits,
            cora_data.y,
            cora_data.test_indices,
        )

    train_accuracy =
        classification_accuracy(
            output.logits,
            cora_data.y,
            cora_data.train_indices,
        )

    validation_accuracy =
        classification_accuracy(
            output.logits,
            cora_data.y,
            cora_data.validation_indices,
        )

    test_accuracy =
        classification_accuracy(
            output.logits,
            cora_data.y,
            cora_data.test_indices,
        )

    return (
        output = output,
        state = new_state,
        train_loss = train_loss,
        validation_loss = validation_loss,
        test_loss = test_loss,
        train_accuracy = train_accuracy,
        validation_accuracy =
            validation_accuracy,
        test_accuracy = test_accuracy,
    )
end

function print_evaluation(
    evaluation;
    prefix = "",
)
    if !isempty(prefix)
        println(prefix)
    end

    println(
        "  Training loss: ",
        round(
            evaluation.train_loss;
            digits = 5,
        ),
    )

    println(
        "  Validation loss: ",
        round(
            evaluation.validation_loss;
            digits = 5,
        ),
    )

    println(
        "  Test loss: ",
        round(
            evaluation.test_loss;
            digits = 5,
        ),
    )

    println(
        "  Training accuracy: ",
        round(
            100 *
            evaluation.train_accuracy;
            digits = 2,
        ),
        "%",
    )

    println(
        "  Validation accuracy: ",
        round(
            100 *
            evaluation.validation_accuracy;
            digits = 2,
        ),
        "%",
    )

    println(
        "  Test accuracy: ",
        round(
            100 *
            evaluation.test_accuracy;
            digits = 2,
        ),
        "%",
    )
end
function copy_parameters(ps)
    return (
        directed_layer = (
            W_vertex =
                copy(ps.directed_layer.W_vertex),

            b_vertex =
                copy(ps.directed_layer.b_vertex),

            W_hyperedge =
                copy(ps.directed_layer.W_hyperedge),

            b_hyperedge =
                copy(ps.directed_layer.b_hyperedge),

            W_vertex_update =
                copy(
                    ps.directed_layer.W_vertex_update
                ),

            b_vertex_update =
                copy(
                    ps.directed_layer.b_vertex_update
                ),
        ),

        W_output =
            copy(ps.W_output),

        b_output =
            copy(ps.b_output),
    )
end


function zero_like_parameters(ps)
    return (
        directed_layer = (
            W_vertex =
                zeros(
                    eltype(
                        ps.directed_layer.W_vertex
                    ),
                    size(
                        ps.directed_layer.W_vertex
                    ),
                ),

            b_vertex =
                zeros(
                    eltype(
                        ps.directed_layer.b_vertex
                    ),
                    size(
                        ps.directed_layer.b_vertex
                    ),
                ),

            W_hyperedge =
                zeros(
                    eltype(
                        ps.directed_layer.W_hyperedge
                    ),
                    size(
                        ps.directed_layer.W_hyperedge
                    ),
                ),

            b_hyperedge =
                zeros(
                    eltype(
                        ps.directed_layer.b_hyperedge
                    ),
                    size(
                        ps.directed_layer.b_hyperedge
                    ),
                ),

            W_vertex_update =
                zeros(
                    eltype(
                        ps.directed_layer.W_vertex_update
                    ),
                    size(
                        ps.directed_layer.W_vertex_update
                    ),
                ),

            b_vertex_update =
                zeros(
                    eltype(
                        ps.directed_layer.b_vertex_update
                    ),
                    size(
                        ps.directed_layer.b_vertex_update
                    ),
                ),
        ),

        W_output =
            zeros(
                eltype(ps.W_output),
                size(ps.W_output),
            ),

        b_output =
            zeros(
                eltype(ps.b_output),
                size(ps.b_output),
            ),
    )
end


function training_objective(
    model,
    ps,
    st,
    X,
    incidence_tail,
    incidence_head,
    labels,
    train_indices,
    weight_decay,
)
    output, _ =
        cora_forward(
            model,
            (
                X,
                incidence_tail,
                incidence_head,
            ),
            ps,
            st,
        )

    data_loss =
        classification_loss(
            output.logits,
            labels,
            train_indices,
        )

    regularisation =
        sum(
            abs2,
            ps.directed_layer.W_vertex,
        ) +
        sum(
            abs2,
            ps.directed_layer.W_hyperedge,
        ) +
        sum(
            abs2,
            ps.directed_layer.W_vertex_update,
        ) +
        sum(
            abs2,
            ps.W_output,
        )

    return data_loss +
           weight_decay *
           regularisation
end


function compute_gradients(
    model,
    ps,
    st,
    cora_data,
    weight_decay,
)
    gradient_parameters =
        zero_like_parameters(ps)

    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        training_objective,
        Enzyme.Const(model),
        Enzyme.Duplicated(
            ps,
            gradient_parameters,
        ),
        Enzyme.Const(st),
        Enzyme.Const(cora_data.X),
        Enzyme.Const(
            cora_data.incidence_tail
        ),
        Enzyme.Const(
            cora_data.incidence_head
        ),
        Enzyme.Const(cora_data.y),
        Enzyme.Const(
            cora_data.train_indices
        ),
        Enzyme.Const(weight_decay),
    )

    return gradient_parameters
end


function all_gradients_finite(
    gradients,
)
    arrays = (
        gradients.directed_layer.W_vertex,
        gradients.directed_layer.b_vertex,
        gradients.directed_layer.W_hyperedge,
        gradients.directed_layer.b_hyperedge,
        gradients.directed_layer.W_vertex_update,
        gradients.directed_layer.b_vertex_update,
        gradients.W_output,
        gradients.b_output,
    )

    return all(
        array -> all(isfinite, array),
        arrays,
    )
end


function gradient_norm(
    gradients,
)
    squared_norm =
        sum(
            abs2,
            gradients.directed_layer.W_vertex,
        ) +
        sum(
            abs2,
            gradients.directed_layer.b_vertex,
        ) +
        sum(
            abs2,
            gradients.directed_layer.W_hyperedge,
        ) +
        sum(
            abs2,
            gradients.directed_layer.b_hyperedge,
        ) +
        sum(
            abs2,
            gradients.directed_layer.W_vertex_update,
        ) +
        sum(
            abs2,
            gradients.directed_layer.b_vertex_update,
        ) +
        sum(
            abs2,
            gradients.W_output,
        ) +
        sum(
            abs2,
            gradients.b_output,
        )

    return sqrt(squared_norm)
end


function train_cora_model(
    model,
    ps,
    st,
    cora_data;
    epochs = NUMBER_OF_EPOCHS,
    learning_rate = LEARNING_RATE,
    weight_decay = 1.0f-5,
)
    epochs > 0 ||
        throw(
            ArgumentError(
                "`epochs` must be positive."
            )
        )

    learning_rate > 0 ||
        throw(
            ArgumentError(
                "`learning_rate` must be positive."
            )
        )

    weight_decay >= 0 ||
        throw(
            ArgumentError(
                "`weight_decay` cannot be negative."
            )
        )

    println()
    println("Training configuration")
    println("Epochs: ", epochs)
    println(
        "Learning rate: ",
        learning_rate,
    )
    println(
        "Weight decay: ",
        weight_decay,
    )
    println(
        "Optimiser: Adam"
    )
    println(
        "Automatic differentiation: Enzyme"
    )

    optimiser =
        Optimisers.Adam(
            learning_rate,
        )

    optimiser_state =
        Optimisers.setup(
            optimiser,
            ps,
        )

    current_ps =
        ps

    current_st =
        st

    best_ps =
        copy_parameters(
            current_ps,
        )

    best_validation_loss =
        Inf

    best_validation_accuracy =
        0.0

    best_epoch =
        0

    training_losses =
        Float64[]

    validation_losses =
        Float64[]

    training_accuracies =
        Float64[]

    validation_accuracies =
        Float64[]

    gradient_norms =
        Float64[]

    println()
    println("Beginning model training...")

    for epoch in 1:epochs

        gradients =
            compute_gradients(
                model,
                current_ps,
                current_st,
                cora_data,
                weight_decay,
            )

        all_gradients_finite(
            gradients
        ) || error(
            "Non-finite gradient detected at epoch $epoch."
        )

        current_gradient_norm =
            gradient_norm(
                gradients,
            )

        push!(
            gradient_norms,
            Float64(
                current_gradient_norm
            ),
        )

        optimiser_state,
        current_ps =
            Optimisers.update(
                optimiser_state,
                current_ps,
                gradients,
            )

        evaluation =
            evaluate_model(
                model,
                current_ps,
                current_st,
                cora_data,
            )

        current_st =
            evaluation.state

        train_loss =
            Float64(
                evaluation.train_loss
            )

        validation_loss =
            Float64(
                evaluation.validation_loss
            )

        train_accuracy =
            Float64(
                evaluation.train_accuracy
            )

        validation_accuracy =
            Float64(
                evaluation.validation_accuracy
            )

        push!(
            training_losses,
            train_loss,
        )

        push!(
            validation_losses,
            validation_loss,
        )

        push!(
            training_accuracies,
            train_accuracy,
        )

        push!(
            validation_accuracies,
            validation_accuracy,
        )

        if validation_loss <
           best_validation_loss

            best_validation_loss =
                validation_loss

            best_validation_accuracy =
                validation_accuracy

            best_epoch =
                epoch

            best_ps =
                copy_parameters(
                    current_ps,
                )
        end

        if epoch == 1 ||
           epoch % 10 == 0 ||
           epoch == epochs

            println(
                "Epoch ",
                lpad(
                    string(epoch),
                    length(
                        string(epochs)
                    ),
                ),
                "/",
                epochs,
                " | train loss = ",
                round(
                    train_loss;
                    digits = 4,
                ),
                " | val loss = ",
                round(
                    validation_loss;
                    digits = 4,
                ),
                " | train acc = ",
                round(
                    100 * train_accuracy;
                    digits = 2,
                ),
                "% | val acc = ",
                round(
                    100 *
                    validation_accuracy;
                    digits = 2,
                ),
                "% | grad norm = ",
                round(
                    current_gradient_norm;
                    digits = 4,
                ),
            )
        end
    end

    println()
    println("Training complete.")

    println(
        "Best epoch: ",
        best_epoch,
    )

    println(
        "Best validation loss: ",
        round(
            best_validation_loss;
            digits = 5,
        ),
    )

    println(
        "Validation accuracy at best epoch: ",
        round(
            100 *
            best_validation_accuracy;
            digits = 2,
        ),
        "%",
    )

    history = (
        training_losses =
            training_losses,

        validation_losses =
            validation_losses,

        training_accuracies =
            training_accuracies,

        validation_accuracies =
            validation_accuracies,

        gradient_norms =
            gradient_norms,

        best_epoch =
            best_epoch,

        best_validation_loss =
            best_validation_loss,

        best_validation_accuracy =
            best_validation_accuracy,
    )

    return (
        parameters = best_ps,
        state = current_st,
        history = history,
    )
end


function print_training_history_summary(
    history,
)
    println()
    println("Training history summary")

    println(
        "Initial training loss: ",
        round(
            first(
                history.training_losses
            );
            digits = 5,
        ),
    )

    println(
        "Final training loss: ",
        round(
            last(
                history.training_losses
            );
            digits = 5,
        ),
    )

    println(
        "Initial validation loss: ",
        round(
            first(
                history.validation_losses
            );
            digits = 5,
        ),
    )

    println(
        "Final validation loss: ",
        round(
            last(
                history.validation_losses
            );
            digits = 5,
        ),
    )

    println(
        "Initial training accuracy: ",
        round(
            100 *
            first(
                history.training_accuracies
            );
            digits = 2,
        ),
        "%",
    )

    println(
        "Final training accuracy: ",
        round(
            100 *
            last(
                history.training_accuracies
            );
            digits = 2,
        ),
        "%",
    )

    println(
        "Initial validation accuracy: ",
        round(
            100 *
            first(
                history.validation_accuracies
            );
            digits = 2,
        ),
        "%",
    )

    println(
        "Final validation accuracy: ",
        round(
            100 *
            last(
                history.validation_accuracies
            );
            digits = 2,
        ),
        "%",
    )

    println(
        "Best epoch: ",
        history.best_epoch,
    )
end
function run_cora_experiment(
    cora_data;
    hidden_dim = HIDDEN_DIM,
    epochs = NUMBER_OF_EPOCHS,
    learning_rate = LEARNING_RATE,
    weight_decay = 1.0f-5,
    seed = RANDOM_SEED,
)
    println()
    println(
        "Cora Directed Hypergraph Neural Network"
    )

    rng =
        MersenneTwister(seed)

    model =
        CoraDirectedClassifier(
            cora_data.number_of_features,
            hidden_dim,
            cora_data.number_of_classes;
            activation = tanh,
            normalize = true,
        )

    ps, st =
        Lux.setup(
            rng,
            model,
        )

    model_summary(
        model,
        ps,
        st,
        cora_data,
    )

    input = (
        cora_data.X,
        cora_data.incidence_tail,
        cora_data.incidence_head,
    )

    println()
    println("Testing initial forward pass...")

    initial_output,
    initial_state =
        cora_forward(
            model,
            input,
            ps,
            st,
        )

    println(
        "Vertex representation size: ",
        size(
            initial_output.hidden_vertices
        ),
    )

    println(
        "Hyperedge representation size: ",
        size(
            initial_output.hidden_hyperedges
        ),
    )

    println(
        "Output logits size: ",
        size(
            initial_output.logits
        ),
    )

    expected_vertex_size = (
        cora_data.number_of_vertices,
        hidden_dim,
    )

    expected_hyperedge_size = (
        cora_data.number_of_hyperedges,
        hidden_dim,
    )

    expected_logit_size = (
        cora_data.number_of_vertices,
        cora_data.number_of_classes,
    )

    size(
        initial_output.hidden_vertices
    ) == expected_vertex_size ||
        error(
            "Unexpected vertex representation size."
        )

    size(
        initial_output.hidden_hyperedges
    ) == expected_hyperedge_size ||
        error(
            "Unexpected hyperedge representation size."
        )

    size(
        initial_output.logits
    ) == expected_logit_size ||
        error(
            "Unexpected classifier output size."
        )

    all(
        isfinite,
        initial_output.hidden_vertices,
    ) ||
        error(
            "Initial vertex representations contain non-finite values."
        )

    all(
        isfinite,
        initial_output.hidden_hyperedges,
    ) ||
        error(
            "Initial hyperedge representations contain non-finite values."
        )

    all(
        isfinite,
        initial_output.logits,
    ) ||
        error(
            "Initial logits contain non-finite values."
        )

    println(
        "Initial forward pass successful."
    )

    println()
    println("Initial model performance")

    initial_evaluation =
        evaluate_model(
            model,
            ps,
            initial_state,
            cora_data,
        )

    print_evaluation(
        initial_evaluation,
    )

    println()
    println(
        "Checking Enzyme gradient computation..."
    )

    initial_gradients =
        compute_gradients(
            model,
            ps,
            initial_state,
            cora_data,
            weight_decay,
        )

    all_gradients_finite(
        initial_gradients
    ) ||
        error(
            "Initial gradient check produced non-finite values."
        )

    initial_gradient_norm =
        gradient_norm(
            initial_gradients,
        )

    println(
        "Initial gradient norm: ",
        round(
            initial_gradient_norm;
            digits = 6,
        ),
    )

    isfinite(
        initial_gradient_norm
    ) ||
        error(
            "Initial gradient norm is not finite."
        )

    println(
        "Enzyme gradient check successful."
    )

    training_result =
        train_cora_model(
            model,
            ps,
            initial_state,
            cora_data;
            epochs = epochs,
            learning_rate = learning_rate,
            weight_decay = weight_decay,
        )

    best_ps =
        training_result.parameters

    final_st =
        training_result.state

    history =
        training_result.history

    print_training_history_summary(
        history,
    )

    println()
    println(
        "Evaluating best validation model..."
    )

    final_evaluation =
        evaluate_model(
            model,
            best_ps,
            final_st,
            cora_data,
        )

    println()
    println("Final model performance")

    print_evaluation(
        final_evaluation,
    )

    test_confusion_matrix =
        confusion_matrix(
            final_evaluation.output.logits,
            cora_data.y,
            cora_data.test_indices,
            cora_data.number_of_classes,
        )

    print_confusion_matrix(
        test_confusion_matrix,
        cora_data.class_names,
    )

    class_results =
        per_class_accuracy(
            final_evaluation.output.logits,
            cora_data.y,
            cora_data.test_indices,
            cora_data.class_names,
        )

    print_per_class_accuracy(
        class_results,
    )

    print_prediction_examples(
        final_evaluation.output.logits,
        cora_data.y,
        cora_data.test_indices,
        cora_data.paper_ids,
        cora_data.class_names;
        number_to_show = 15,
    )

    predictions =
        predicted_classes(
            final_evaluation.output.logits,
        )

    println()
    println(
        "Proof-of-concept experiment summary"
    )

    println(
        "Dataset: Cora citation network"
    )

    println(
        "Number of papers: ",
        cora_data.number_of_vertices,
    )

    println(
        "Number of directed citation hyperedges: ",
        cora_data.number_of_hyperedges,
    )

    println(
        "Input features per paper: ",
        cora_data.number_of_features,
    )

    println(
        "Number of classes: ",
        cora_data.number_of_classes,
    )

    println(
        "Hidden representation dimension: ",
        hidden_dim,
    )

    println(
        "Training nodes: ",
        length(
            cora_data.train_indices
        ),
    )

    println(
        "Validation nodes: ",
        length(
            cora_data.validation_indices
        ),
    )

    println(
        "Test nodes: ",
        length(
            cora_data.test_indices
        ),
    )

    println(
        "Best training epoch: ",
        history.best_epoch,
    )

    println(
        "Final training accuracy: ",
        round(
            100 *
            final_evaluation.train_accuracy;
            digits = 2,
        ),
        "%",
    )

    println(
        "Final validation accuracy: ",
        round(
            100 *
            final_evaluation.validation_accuracy;
            digits = 2,
        ),
        "%",
    )

    println(
        "Final test accuracy: ",
        round(
            100 *
            final_evaluation.test_accuracy;
            digits = 2,
        ),
        "%",
    )

    println(
        "Final training loss: ",
        round(
            final_evaluation.train_loss;
            digits = 5,
        ),
    )

    println(
        "Final validation loss: ",
        round(
            final_evaluation.validation_loss;
            digits = 5,
        ),
    )

    println(
        "Final test loss: ",
        round(
            final_evaluation.test_loss;
            digits = 5,
        ),
    )

    println()
    println(
        "The proof of concept demonstrates an end-to-end"
    )

    println(
        "directed hypergraph node-classification pipeline"
    )

    println(
        "using the HyperGraphNeuralNetworks.jl package."
    )

    return (
        model = model,
        parameters = best_ps,
        state = final_st,
        history = history,
        evaluation = final_evaluation,
        predictions = predictions,
        confusion_matrix =
            test_confusion_matrix,
        per_class_results =
            class_results,
    )
end


function main()
    println()
    println(
        "Cora Node Classification Proof of Concept"
    )

    println(
        "Random seed: ",
        RANDOM_SEED,
    )

    cora_data =
        prepare_cora()

    experiment =
        run_cora_experiment(
            cora_data;
            hidden_dim = HIDDEN_DIM,
            epochs = NUMBER_OF_EPOCHS,
            learning_rate = LEARNING_RATE,
            weight_decay = 1.0f-5,
            seed = RANDOM_SEED,
        )

    println()
    println(
        "Cora node classification experiment finished successfully."
    )

    return (
        data = cora_data,
        experiment = experiment,
    )
end


if abspath(PROGRAM_FILE) == @__FILE__
    CORA_RESULTS =
        main()
end
