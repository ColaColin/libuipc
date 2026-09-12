// accuracy vs cond(F): fixed-sweep(4) Jacobi vs the shipped iterative QR-SVD
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
int main(int argc,char**argv){
    long N=(argc>1)?atol(argv[1]):20000;
    std::mt19937_64 rng(7);
    std::normal_distribution<double> g(0,1);
    auto randR=[&](){M3 A;for(int i=0;i<3;++i)for(int j=0;j<3;++j)A(i,j)=g(rng);
        Eigen::HouseholderQR<M3> qr(A);M3 R=qr.householderQ();
        if(R.determinant()<0)R.col(2)=-R.col(2);return R;};
    printf("%-12s %-12s %-12s %-12s %-12s\n","cond(F)","new recon","old recon","new dS/|F|","old dS/|F|");
    for(double c : {1.0,1e1,1e2,1e3,1e4,1e5,1e6,1e8,1e10}){
        double mn=0,mo=0,dn=0,do_=0;
        for(long k=0;k<N;++k){
            V3 sv; sv<<1.0,1.0/std::sqrt(c),1.0/c;
            M3 R1=randR(),R2=randR();
            M3 F=R1*sv.asDiagonal()*R2.transpose();
            double sc=std::max(F.cwiseAbs().maxCoeff(),1e-300);
            M3 U,V,Uo,Vo; V3 S,So;
            math::qr_svd_fixed<4>(F,S,U,V);
            math::qr_svd(F,So,Uo,Vo);
            mn=std::max(mn,(U*S.asDiagonal()*V.transpose()-F).cwiseAbs().maxCoeff()/sc);
            mo=std::max(mo,(Uo*So.asDiagonal()*Vo.transpose()-F).cwiseAbs().maxCoeff()/sc);
            // singular values against the exact ones we constructed
            V3 ex; ex<<sv(0),sv(1),sv(2);
            dn=std::max(dn,(S-ex).cwiseAbs().maxCoeff()/sc);
            do_=std::max(do_,(So-ex).cwiseAbs().maxCoeff()/sc);
        }
        printf("%-12.0e %-12.3e %-12.3e %-12.3e %-12.3e\n",c,mn,mo,dn,do_);
    }
    return 0;
}
