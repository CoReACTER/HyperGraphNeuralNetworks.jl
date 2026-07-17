# directed hypergraph regression model

#combines a DirectedHypergraphLayer with a Lux Dense regression head.
#the model produces one continuous prediction for every reaction/hyperedge.


struct DirectedHypergraphRegression{H, R} <:
       Lux.AbstractLuxContainerLayer{(:hypergraph_layer, :regression_head)}

    hypergraph_layer::H
    regression_head::R
end


"""
    DirectedHypergraphRegression(
        species_in_dim,
        hidden_dim;
        activation = tanh
    )

Construct a regression model that combines a
`DirectedHypergraphLayer` with a Lux `Dense` regression head.

The model performs:

species features
→ directed hypergraph message passing
→ reaction embeddings
→ dense regression head
→ one scalar prediction per reaction

# Arguments

- `species_in_dim`: Number of input features for each species.
- `hidden_dim`: Size of the hidden embeddings.
- `activation`: Activation function used in the hypergraph layer.
"""
function DirectedHypergraphRegression(
    species_in_dim::Int,
    hidden_dim::Int;
    activation = tanh
)
    hypergraph_layer = DirectedHypergraphLayer(
        species_in_dim,
        hidden_dim,
        activation
    )

    regression_head = Lux.Dense(hidden_dim => 1)

    return DirectedHypergraphRegression(
        hypergraph_layer,
        regression_head
    )
end


# forward pass
"""
    (model::DirectedHypergraphRegression)(input, ps, st)

Run a forward pass through the regression model.

The model first computes reaction embeddings using the
`DirectedHypergraphLayer` and then applies a Dense regression head to
produce one scalar prediction for each reaction.

# Arguments

- `input`: Tuple `(X_species, source_matrix, target_matrix)`
- `ps`: Lux parameters
- `st`: Lux state

# Returns

A tuple containing:

- `output`, a named tuple with:
    - `predictions`
    - `reaction_embeddings`
    - `updated_species`
- `new_state`, the updated Lux state.
"""

function (model::DirectedHypergraphRegression)(input, ps, st)
    X_species, source_matrix, target_matrix = input

    # directed hypergraph message passing

    hypergraph_output, hypergraph_state = model.hypergraph_layer(
        (
            X_species,
            source_matrix,
            target_matrix
        ),
        ps.hypergraph_layer,
        st.hypergraph_layer
    )

    reaction_embeddings =
        hypergraph_output.reaction_embeddings

    # Lux Dense expects features × batch.
    # The reaction embeddings currently have shape:
    # reactions × hidden features.

    reaction_embeddings_for_dense =
        transpose(reaction_embeddings)

    prediction_matrix, regression_state = model.regression_head(
        reaction_embeddings_for_dense,
        ps.regression_head,
        st.regression_head
    )

    # Convert the 1 × number_of_reactions matrix
    # into a vector containing one prediction per reaction.

    predictions = vec(prediction_matrix)

    output = (
        predictions = predictions,
        reaction_embeddings = reaction_embeddings,
        updated_species = hypergraph_output.updated_species
    )

    new_state = (
        hypergraph_layer = hypergraph_state,
        regression_head = regression_state
    )

    return output, new_state
end