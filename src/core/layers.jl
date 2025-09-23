
using Random

"""
    HGNNConv(in_ft::Int, out_ft::Int; bias::Bool=true)
"""

"""
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x
"""
# weight is different: what type of data we use in lux machine learning

struct HGNNConv{T: Real} <: GNNLayer{T}
    in_ft::Int
    out_ft::Int
    weight::AbstractArray{T, 2} # i'm not sure but we want dimention od in_ft*out_ft                      
    bias:: Union{Nothing, Vector{T}}    
end

function HGNN_conv{T}(
    in_ft::Int, 
    out_ft::Int; 
    weight::Union{AbstractArray{}, Nothing} = nothing,
    initialBias::Union{Vector{T}, Nothing} = nothing,
    use_bias::Bool = true
) where {T<:Real}

    if weight !== nothing
        @assert size(weight) == (in_ft, out_ft) "Weight matrix must be of size (in_ft, out_ft)"
    else
        stdv = 1. / sqrt(out_ft) 
        lo, hi = -stdv, stdv
        weight = lo .+ (hi-lo) .* rand(T, in_ft, out_ft)
    end

    if use_bias === false
        bias = vector{Nothing}(out_ft, 1)
    else
        if initialbias === nothing
            bias = lo .+ (hi-lo) .* rand(T, out_ft, 1)
        else
        @assert length(initialBias) == out_ft "Bias vector must be of length out_ft"
        bias = initialBias
        end
    end

    HGNN_conv{T}(
        weight = weight,
        bias = bias,
    )
end

function forward(HGNNConv::HGNNConv{T}, x::Matrix{T}, G::Matrix{T}) where {T<:Real}
    x = x * HGNNConv.weight
    if HGNNConv.bias !== nothing
        x = x .+ HGNNConv.bias
    end
    x = G * x
    return x
end
