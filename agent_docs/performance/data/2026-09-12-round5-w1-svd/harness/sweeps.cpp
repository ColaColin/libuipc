// How many Jacobi sweeps does each sample actually need? (p01-style histogram)
// "needs k sweeps" = the smallest k whose reconstruction error is <= 1e-14*max|F|.
#include <algorithm/qr_svd.hpp>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
namespace uipc::backend::cuda { namespace math {
#include "svd_fixed_body.hpp"
} }
using namespace uipc::backend::cuda;
using M3 = Eigen::Matrix<double,3,3>;
using V3 = Eigen::Matrix<double,3,1>;

template<int NS> static double err(const M3& F){
    M3 U,V; V3 S; math::qr_svd_fixed<NS>(F,S,U,V);
    double sc = std::max(F.cwiseAbs().maxCoeff(),1e-300);
    return (U*S.asDiagonal()*V.transpose()-F).cwiseAbs().maxCoeff()/sc;
}
static int need(const M3& F){
    const double tol = 1e-14;
    if(err<1>(F)<=tol) return 1;
    if(err<2>(F)<=tol) return 2;
    if(err<3>(F)<=tol) return 3;
    if(err<4>(F)<=tol) return 4;
    if(err<5>(F)<=tol) return 5;
    if(err<6>(F)<=tol) return 6;
    return 7;
}
static void run(const char* name, std::vector<M3>& Fs){
    std::vector<int> it(Fs.size());
    for(size_t i=0;i<Fs.size();++i) it[i]=need(Fs[i]);
    double mean=0; for(int v:it) mean+=v; mean/=it.size();
    std::vector<int> s=it; std::sort(s.begin(),s.end());
    int hist[9]={0}; for(int v:it) hist[std::min(v,8)]++;
    printf("%-28s n=%zu mean=%.2f median=%d p95=%d max=%d   hist[1..6]=%d/%d/%d/%d/%d/%d  >6:%d\n",
        name,it.size(),mean,s[s.size()/2],s[(size_t)(s.size()*0.95)],s.back(),
        hist[1],hist[2],hist[3],hist[4],hist[5],hist[6],hist[7]+hist[8]);
}
int main(int argc,char**argv){
    long N=(argc>1)?atol(argv[1]):65536;
    std::mt19937_64 rng(20260912);
    std::normal_distribution<double> g(0,1);
    auto randM=[&](M3&M){for(int i=0;i<3;++i)for(int j=0;j<3;++j)M(i,j)=g(rng);};
    auto randR=[&](){M3 A;randM(A);Eigen::HouseholderQR<M3> qr(A);M3 R=qr.householderQ();
                     if(R.determinant()<0)R.col(2)=-R.col(2);return R;};
    for(double eps:{1e-4,1e-2,1e-1,3e-1}){
        std::vector<M3> Fs(N); for(auto&F:Fs){M3 Nn;randM(Nn);F=M3::Identity()+eps*Nn;}
        char b[64];snprintf(b,64,"F = I + %.0e*N(0,1)",eps);run(b,Fs);
    }
    {std::vector<M3> Fs(N);for(auto&F:Fs)randM(F);run("F = full random N(0,1)",Fs);}
    {std::vector<M3> Fs(N);for(auto&F:Fs){M3 Nn;randM(Nn);F=randR()*(M3::Identity()+1e-2*Nn);}
     run("F = R*(I+1e-2*N)",Fs);}
    {std::vector<M3> Fs(N);V3 sv;for(auto&F:Fs){double a=std::exp(g(rng)),b2=std::exp(g(rng));
        sv<<std::max(a,b2),std::min(a,b2),std::min(a,b2);F=randR()*sv.asDiagonal()*randR().transpose();}
     run("2 equal singular values",Fs);}
    {std::vector<M3> Fs(N);V3 sv;for(auto&F:Fs){double a=std::exp(g(rng));sv<<a,a,a;
        F=randR()*sv.asDiagonal()*randR().transpose();} run("3 equal singular values",Fs);}
    {std::vector<M3> Fs(N);V3 sv;for(auto&F:Fs){sv<<1.0,0.5,0.0;
        F=randR()*sv.asDiagonal()*randR().transpose();} run("rank 2 (s2 = 0)",Fs);}
    {std::vector<M3> Fs(N);for(auto&F:Fs){randM(F);if(F.determinant()>0)F.col(0)=-F.col(0);}
     run("reflection (det < 0)",Fs);}
    for(double cond:{1e2,1e4,1e6}){
        std::vector<M3> Fs(N);V3 sv;for(auto&F:Fs){sv<<1.0,1.0/std::sqrt(cond),1.0/cond;
            F=randR()*sv.asDiagonal()*randR().transpose();}
        char b[64];snprintf(b,64,"cond(F) = %.0e",cond);run(b,Fs);
    }
    return 0;
}
