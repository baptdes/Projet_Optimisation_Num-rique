# Ecrire les tests de l'algorithme du pas de Cauchy
using Test

function tester_cauchy(cauchy::Function)

    Test.@testset "Pas de Cauchy" begin

        g = [0, 0]
        H = [7 0; 0 2]
        Δ = 1
        s = cauchy(g, H, Δ)
        Test.@test isapprox(s, [0.0, 0.0], atol=1e-10)

        g = [1, 1]
        H = [1 1; 1 1]
        Δ = 1
        s = cauchy(g, H, Δ)
        Test.@test isapprox(s, [0.5, 0.5], atol=1e-10)

        #TODO: Ajouter des tests
    end

end