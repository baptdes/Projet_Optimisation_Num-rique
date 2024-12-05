# Tests de l'algorithme du pas de Cauchy
using Test

function tester_cauchy(cauchy::Function)

    Test.@testset "Pas de Cauchy" begin

        @testset "Cas a = 0" begin
            # Cas où a = 0 & b = 0
            g = [0, 0]
            H = [7 0; 0 2]
            Δ = 1
            s = cauchy(g, H, Δ)
            Test.@test isapprox(s, [0.0, 0.0], atol=1e-10)

            # Cas où a = 0 & b ≠ 0
            g = [4; 0]
            H = [0 0;
                  0 0]
            Δ = 8
            s = cauchy(g,H,Δ)
            Test.@test isapprox(s, - [8.0, 0.0], atol=1e-10)
        end

        @testset "Cas a < 0" begin
            g = [12; 5]
            H = - [1 0; 0 1]
            Δ = 1.0
            s = cauchy(g,H,Δ)

            @test isapprox(s,-Δ/norm(g)*g, atol=1e-10)
        end

        @testset "Cas a > 0" begin

            # Cas où t = Δ / ‖g‖
            g = [1, 1] 
            H = - [1 1; 1 1]
            Δ = 0.1
            s = cauchy(g, H, Δ)

            a = g' * H * g
            b = norm(g)^2
            Test.@test isapprox(s, -Δ/norm(g)*g, atol=1e-10)

            # Cas où t = -b / a
            g = [1, 1]
            H = [1 1; 1 1]
            Δ = 1
            s = cauchy(g, H, Δ)

            a = g' * H * g
            b = norm(g)^2
            Test.@test isapprox(s, (b / a) * g, atol=1e-10)

            g = [1, 1]
            H = [1 1; 1 1]
            Δ = 1
            s = cauchy(g, H, Δ)
            Test.@test isapprox(s, [0.5, 0.5], atol=1e-10)
        end
    end

end