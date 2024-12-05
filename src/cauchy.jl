using LinearAlgebra
"""
Approximation de la solution du problème 

    min qₖ(s) = s'gₖ + 1/2 s' Hₖ s

        sous les contraintes s = -t gₖ, t > 0, ‖s‖ ≤ Δₖ

# Syntaxe

    s = cauchy(g, H, Δ; kwargs...)

# Entrées

    - g : (Vector{<:Real}) le vecteur gₖ
    - H : (Matrix{<:Real}) la matrice Hₖ
    - Δ : (Real) le scalaire Δₖ
    - kwargs  : les options sous formes d'arguments "keywords", c'est-à-dire des arguments nommés
        • tol_abs  : la tolérence absolue (optionnel, par défaut 1e-10)

# Sorties

    - s : (Vector{<:Real}) la solution du problème

# Exemple d'appel

    g = [0; 0]
    H = [7 0 ; 0 2]
    Δ = 1
    s = cauchy(g, H, Δ)

"""
function cauchy(g::Vector{<:Real}, H::Matrix{<:Real}, Δ::Real; tol_abs::Real = 1e-10)

    s = zero(length(g))
    norm_g = norm(g)

    if norm_g > tol_abs
        t_etoile = (norm_g^2)/(g'*H*g)
        t_max = Δ/norm_g

        if abs(t_etoile) < t_max
            s = -t_etoile*g
        else
            s = -t_max*g
        end
    end

    return s
end

function old_cauchy(g::Vector{<:Real}, H::Matrix{<:Real}, Δ::Real; tol_abs::Real = 1e-10)
    g_norm = norm(g)
    a = g' * H * g
    b = g_norm^2

    if(g_norm <= tol_abs)
        return zeros(length(g))
    end

    if (a <= 0)
        t = Δ / g_norm
    else
        t = min(Δ / g_norm, - b / a)
    end
    
    return -t * g
end