// Randomised-input verifier: fixed-sweep Jacobi SVD vs the shipped iterative
// Ziran QR-SVD (algorithm/qr_svd.hpp), host build of the same UIPC_GENERIC code.
#include <algorithm/qr_svd.hpp>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <string>

namespace uipc::backend::cuda { namespace math {
#include "svd_fixed_body.hpp"
} }

using namespace uipc::backend::cuda;
using M3 = Eigen::Matrix<double,3,3>;
using V3 = Eigen::Matrix<double,3,1>;

struct Stat {
    double max_recon=0, max_orthU=0, max_orthV=0, max_detU=0, max_detV=0;
    double max_sdiff=0, max_ord=0, max_signerr=0;
    double sum_recon=0; long n=0;
    void add(double recon,double ou,double ov,double du,double dv,double sd,double ord,double se){
        max_recon=std::max(max_recon,recon); max_orthU=std::max(max_orthU,ou);
        max_orthV=std::max(max_orthV,ov); max_detU=std::max(max_detU,du);
        max_detV=std::max(max_detV,dv); max_sdiff=std::max(max_sdiff,sd);
        max_ord=std::max(max_ord,ord); max_signerr=std::max(max_signerr,se);
        sum_recon+=recon; ++n;
    }
};

template<int NS>
static void check(const M3& F, Stat& st, Stat& stold)
{
    M3 U,V,Uo,Vo; V3 S,So;
    math::qr_svd_fixed<NS>(F,S,U,V);
    math::qr_svd(F,So,Uo,Vo);

    const double sc = std::max(F.cwiseAbs().maxCoeff(), 1e-300);
    auto eval=[&](const M3&U_,const V3&S_,const M3&V_,Stat& s_){
        double recon = (U_*S_.asDiagonal()*V_.transpose() - F).cwiseAbs().maxCoeff()/sc;
        double ou = (U_.transpose()*U_ - M3::Identity()).cwiseAbs().maxCoeff();
        double ov = (V_.transpose()*V_ - M3::Identity()).cwiseAbs().maxCoeff();
        double du = std::abs(U_.determinant()-1.0);
        double dv = std::abs(V_.determinant()-1.0);
        // ordering: S0 >= S1 >= |S2|
        double ord = std::max(0.0, std::max(S_(1)-S_(0), std::abs(S_(2))-S_(1)))/sc;
        // sign of S2 must match sign(det F)
        double detF = F.determinant();
        double se = 0;
        if(std::abs(detF) > 1e-12*sc*sc*sc)
            se = (std::signbit(S_(2)) == std::signbit(detF)) ? 0.0 : 1.0;
        s_.add(recon,ou,ov,du,dv,0,ord,se);
    };
    eval(U,S,V,st);
    eval(Uo,So,Vo,stold);
    // singular values: compare new vs old (they are a well-conditioned quantity)
    double sd = (S-So).cwiseAbs().maxCoeff()/sc;
    st.max_sdiff = std::max(st.max_sdiff, sd);
}

static void report(const char* tag,const char* name, const Stat& s){
    printf("%-8s %-34s n=%-8ld recon=%.3e sdiff=%.3e U^TU=%.3e V^TV=%.3e |detU-1|=%.3e |detV-1|=%.3e ord=%.3e signerr=%.0f\n",
        tag,name,s.n,s.max_recon,s.max_sdiff,s.max_orthU,s.max_orthV,s.max_detU,s.max_detV,s.max_ord,s.max_signerr);
}

template<int NS>
static void run_all(long N)
{
    std::mt19937_64 rng(20260912);
    std::normal_distribution<double> g(0,1);
    std::uniform_real_distribution<double> uni(0,1);
    auto randM=[&](M3&M){ for(int i=0;i<3;++i)for(int j=0;j<3;++j) M(i,j)=g(rng); };
    auto randR=[&](){ M3 A; randM(A); Eigen::HouseholderQR<M3> qr(A); M3 R=qr.householderQ();
                      if(R.determinant()<0) R.col(2)=-R.col(2); return R; };

    struct Case { const char* name; };
    const char* names[] = {
      "F = I (exact)",
      "F = I + 1e-8*N",
      "F = I + 1e-4*N",
      "F = I + 1e-1*N",
      "F = full random N(0,1)",
      "F = R*(I+1e-2*N)  (near-rigid)",
      "repeated sing. vals (2 equal)",
      "repeated sing. vals (3 equal)",
      "near-zero det (s2 ~ 1e-12)",
      "exactly rank 2 (s2 = 0)",
      "exactly rank 1 (s1=s2=0)",
      "rank 0 (F = 0)",
      "reflection (det < 0)",
      "ill-conditioned (cond 1e10)",
      "ill-conditioned (cond 1e14)",
      "huge scale (1e150)",
      "tiny scale (1e-150)",
      "tet inversion (s2 -> -s2)",
    };
    const int NC = sizeof(names)/sizeof(names[0]);
    for(int ci=0; ci<NC; ++ci)
    {
        Stat st, sto;
        for(long k=0;k<N;++k)
        {
            M3 F;
            M3 R1=randR(), R2=randR();
            V3 sv;
            switch(ci){
            case 0: F=M3::Identity(); break;
            case 1: randM(F); F=M3::Identity()+1e-8*F; break;
            case 2: randM(F); F=M3::Identity()+1e-4*F; break;
            case 3: randM(F); F=M3::Identity()+1e-1*F; break;
            case 4: randM(F); break;
            case 5: { M3 N_; randM(N_); F=randR()*(M3::Identity()+1e-2*N_); } break;
            case 6: { double a=std::exp(g(rng)); double b=std::exp(g(rng));
                      sv<<std::max(a,b),std::min(a,b),std::min(a,b); F=R1*sv.asDiagonal()*R2.transpose(); } break;
            case 7: { double a=std::exp(g(rng)); sv<<a,a,a; F=R1*sv.asDiagonal()*R2.transpose(); } break;
            case 8: { sv<<1.0,0.5,1e-12*uni(rng); F=R1*sv.asDiagonal()*R2.transpose(); } break;
            case 9: { sv<<1.0,0.5,0.0; F=R1*sv.asDiagonal()*R2.transpose(); } break;
            case 10:{ sv<<1.0,0.0,0.0; F=R1*sv.asDiagonal()*R2.transpose(); } break;
            case 11: F.setZero(); break;
            case 12:{ randM(F); if(F.determinant()>0) F.col(0)=-F.col(0); } break;
            case 13:{ sv<<1.0,1e-5,1e-10; F=R1*sv.asDiagonal()*R2.transpose(); } break;
            case 14:{ sv<<1.0,1e-7,1e-14; F=R1*sv.asDiagonal()*R2.transpose(); } break;
            case 15:{ randM(F); F*=1e150; } break;
            case 16:{ randM(F); F*=1e-150; } break;
            case 17:{ M3 N_; randM(N_); F=randR()*(M3::Identity()+1e-2*N_); F.col(2)=-F.col(2); } break;
            }
            check<NS>(F,st,sto);
            if(ci==0||ci==11) break; // deterministic cases: one sample is enough
        }
        char tag[16]; snprintf(tag,16,"NS=%d",NS);
        report(tag,names[ci],st);
        report("  OLD",names[ci],sto);
    }
}

int main(int argc,char** argv)
{
    long N = (argc>1)? atol(argv[1]) : 100000;
    int only = (argc>2)? atoi(argv[2]) : 0;
    if(only==0||only==2){ printf("=== 2 sweeps ===\n"); run_all<2>(N); }
    if(only==0||only==3){ printf("=== 3 sweeps ===\n"); run_all<3>(N); }
    if(only==0||only==4){ printf("=== 4 sweeps ===\n"); run_all<4>(N); }
    if(only==0||only==5){ printf("=== 5 sweeps ===\n"); run_all<5>(N); }
    if(only==0||only==6){ printf("=== 6 sweeps ===\n"); run_all<6>(N); }
    return 0;
}
