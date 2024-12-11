using LinearAlgebra
"""
Approximation de la solution du problème 

    min qₖ(s) = s'gₖ + 1/2 s' Hₖ s, sous la contrainte ‖s‖ ≤ Δₖ

# Syntaxe

    s = gct(g, H, Δ; kwargs...)

# Entrées

    - g : (Vector{<:Real}) le vecteur gₖ
    - H : (Matrix{<:Real}) la matrice Hₖ
    - Δ : (Real) le scalaire Δₖ
    - kwargs  : les options sous formes d'arguments "keywords", c'est-à-dire des arguments nommés
        • max_iter : le nombre maximal d'iterations (optionnel, par défaut 100)
        • tol_abs  : la tolérence absolue (optionnel, par défaut 1e-10)
        • tol_rel  : la tolérence relative (optionnel, par défaut 1e-8)

# Sorties

    - s : (Vector{<:Real}) une approximation de la solution du problème

# Exemple d'appel

    g = [0; 0]
    H = [7 0 ; 0 2]
    Δ = 1
    s = gct(g, H, Δ)

"""
function gct(g::Vector{<:Real}, H::Matrix{<:Real}, Δ::Real; 
    max_iter::Integer = 100, 
    tol_abs::Real = 1e-10, 
    tol_rel::Real = 1e-8)

    s = zeros(length(g))
    j = 0
    norm_g0 = norm(g);
    gj = g
    pj = -gj
    while j <= max_iter && norm(gj) > max(norm_g0 * tol_rel, tol_abs)

        k = pj' * H * pj

        if k <= 0
            # Calcul des racines de norm(s + sigma * pj) = Δ
            a = norm(pj)^2
            b = 2 * (s' * pj)
            c = norm(s)^2 - Δ^2
            delta = b^2 - 4 * a * c
            sigma1 = (-b + sqrt(delta)) / (2 * a)
            sigma2 = (-b - sqrt(delta)) / (2 * a)

            # Choix de la racine pour laquelle q(sj + σpj) est la plus petite
            function q(s)
                return s' * gj + 0.5 * s' * H * s
            end
            if q(s + sigma1 * pj) <= q(s + sigma2 * pj)
                sigma = sigma1
            else
                sigma = sigma2
            end

            return s + sigma * pj
        end

        alpha = (gj' * gj) / k

        if norm(s + alpha * pj) >= Δ
            # Calcul des racines de norm(s + sigma * pj) = Δ
            a = norm(pj)^2
            b = 2 * (s' * pj)
            c = norm(s)^2 - Δ^2
            delta = b^2 - 4 * a * c
            sigma = (-b + sqrt(delta)) / (2 * a)

            return s + sigma * pj
        end

        s += alpha * pj
        gj1 = gj + alpha * H * pj
        β = (gj1' * gj1) / (gj' * gj)
        pj = -gj1 + β * pj
        gj = gj1
        j += 1
    end
   return s
end
