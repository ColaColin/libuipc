static std::vector<HingeInput> gen_inputs(int n, unsigned seed, int model)
{
    std::mt19937_64              rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<HingeInput>      h(n);
    for(int i = 0; i < n; ++i)
    {
        double L0 = 0.014 + U(rng) * 0.036;
        double th = U(rng) * 2 * M_PI, ph = std::acos(2 * U(rng) - 1);
        Eigen::Vector3d dir(std::sin(ph) * std::cos(th), std::sin(ph) * std::sin(th), std::cos(ph));
        Eigen::Vector3d any(0, 0, 1);
        if(std::abs(dir.z()) > 0.9)
            any = Eigen::Vector3d(1, 0, 0);
        Eigen::Vector3d u = dir.cross(any).normalized();
        Eigen::Vector3d mid_edge = 0.5 * L0 * dir;
        double h0 = (U(rng) < 0.02 ? 1e-4 : 0.004 + U(rng) * 0.026);
        double h3 = (U(rng) < 0.02 ? 1e-4 : 0.004 + U(rng) * 0.026);
        double lat0 = (U(rng) - 0.5) * 0.02, lat3 = (U(rng) - 0.5) * 0.02;
        Eigen::Vector3d x1 = mid_edge, x2 = -mid_edge;
        Eigen::Vector3d x0 = lat0 * dir + h0 * u;
        Eigen::Vector3d x3 = lat3 * dir - h3 * u;
        for(int k = 0; k < 3; ++k)
        {
            h[i].x[0][k] = x0(k);
            h[i].x[1][k] = x1(k);
            h[i].x[2][k] = x2(k);
            h[i].x[3][k] = x3(k);
        }
        h[i].kappa = std::pow(10.0, -6.0 + 6.0 * U(rng));
        h[i].L0 = L0;
        h[i].h_bar = 0.0008 + U(rng) * 0.003;
        h[i].theta_bar = U(rng) < 0.5 ? 0.0 : (U(rng) - 0.5) * 3.0;
        double theta_y = std::pow(10.0, -3.0 + 2.8 * U(rng));
        double slope = 2.0 * h[i].kappa * L0 / h[i].h_bar;
        h[i].yield_p = model == 0 ? -1.0 : theta_y * slope;  // negative tags strain
        h[i].Vdt2 = std::pow(10.0, -9.0 + 3.0 * U(rng));
    }
    return h;
}

