#include <memory.h>
#include <math.h>

#include "sba.h"

#include "sba_warper.h"
#include "global.h"
using global::cout;
//using std::cout;
using std::endl;

/* unit quaternion from vector part */
#define _MK_QUAT_FRM_VEC(q, v){                                     \
	(q)[1] = (v)[0]; (q)[2] = (v)[1]; (q)[3] = (v)[2];                      \
	(q)[0] = sqrt(1.0 - (q)[1] * (q)[1] - (q)[2] * (q)[2] - (q)[3] * (q)[3]);  \
}

/* convert a vector of camera parameters so that rotation is represented by
* the vector part of the input quaternion. The function converts the
* input quaternion into a unit one with a non-negative scalar part. Remaining
* parameters are left unchanged.
*
* Input parameter layout: intrinsics (5, optional), distortion (5, optional), rot. quaternion (4), translation (3)
* Output parameter layout: intrinsics (5, optional), distortion (5, optional), rot. quaternion vector part (3), translation (3)
*/
void quat2vec(double *inp, int nin, double *outp, int nout)
{
	double mag, sg;
	register int i;

	/* intrinsics & distortion */
	if (nin>7) // are they present?
	for (i = 0; i<nin - 7; ++i)
		outp[i] = inp[i];
	else
		i = 0;

	/* rotation */
	/* normalize and ensure that the quaternion's scalar component is non-negative;
	* if not, negate the quaternion since two quaternions q and -q represent the
	* same rotation
	*/
	mag = sqrt(inp[i] * inp[i] + inp[i + 1] * inp[i + 1] + inp[i + 2] * inp[i + 2] + inp[i + 3] * inp[i + 3]);
	sg = (inp[i] >= 0.0) ? 1.0 : -1.0;
	mag = sg / mag;
	outp[i] = inp[i + 1] * mag;
	outp[i + 1] = inp[i + 2] * mag;
	outp[i + 2] = inp[i + 3] * mag;
	i += 3;

	/* translation*/
	for (; i<nout; ++i)
		outp[i] = inp[i + 1];
}

/* convert a vector of camera parameters so that rotation is represented by
* a full unit quaternion instead of its input 3-vector part. Remaining
* parameters are left unchanged.
*
* Input parameter layout: intrinsics (5, optional), distortion (5, optional), rot. quaternion vector part (3), translation (3)
* Output parameter layout: intrinsics (5, optional), distortion (5, optional), rot. quaternion (4), translation (3)
*/
void vec2quat(double *inp, int nin, double *outp, int nout)
{
	double *v, q[FULLQUATSZ];
	register int i;

	/* intrinsics & distortion */
	if (nin>7 - 1) // are they present?
	for (i = 0; i<nin - (7 - 1); ++i)
		outp[i] = inp[i];
	else
		i = 0;

	/* rotation */
	/* recover the quaternion from the vector */
	v = inp + i;
	_MK_QUAT_FRM_VEC(q, v);
	outp[i] = q[0];
	outp[i + 1] = q[1];
	outp[i + 2] = q[2];
	outp[i + 3] = q[3];
	i += FULLQUATSZ;

	/* translation */
	for (; i<nout; ++i)
		outp[i] = inp[i - 1];
}


/*
* fast multiplication of the two quaternions in q1 and q2 into p
* this is the second of the two schemes derived in pg. 8 of
* T. D. Howell, J.-C. Lafon, The complexity of the quaternion product, TR 75-245, Cornell Univ., June 1975.
*
* total additions increase from 12 to 27 (28), but total multiplications decrease from 16 to 9 (12)
*/
inline static void quatMultFast(double q1[FULLQUATSZ], double q2[FULLQUATSZ], double p[FULLQUATSZ])
{
	double t1, t2, t3, t4, t5, t6, t7, t8, t9;
	//double t10, t11, t12;

	t1 = (q1[0] + q1[1])*(q2[0] + q2[1]);
	t2 = (q1[3] - q1[2])*(q2[2] - q2[3]);
	t3 = (q1[1] - q1[0])*(q2[2] + q2[3]);
	t4 = (q1[2] + q1[3])*(q2[1] - q2[0]);
	t5 = (q1[1] + q1[3])*(q2[1] + q2[2]);
	t6 = (q1[1] - q1[3])*(q2[1] - q2[2]);
	t7 = (q1[0] + q1[2])*(q2[0] - q2[3]);
	t8 = (q1[0] - q1[2])*(q2[0] + q2[3]);

#if 0
	t9 = t5 + t6;
	t10 = t7 + t8;
	t11 = t5 - t6;
	t12 = t7 - t8;

	p[0] = t2 + 0.5*(-t9 + t10);
	p[1] = t1 - 0.5*(t9 + t10);
	p[2] = -t3 + 0.5*(t11 + t12);
	p[3] = -t4 + 0.5*(t11 - t12);
#endif

	/* following fragment it equivalent to the one above */
	t9 = 0.5*(t5 - t6 + t7 + t8);
	p[0] = t2 + t9 - t5;
	p[1] = t1 - t9 - t6;
	p[2] = -t3 + t9 - t8;
	p[3] = -t4 + t9 - t7;
}

/* Routines to estimate the estimated measurement vector (i.e. "func") and
* its sparse jacobian (i.e. "fjac") needed in BA. Code below makes use of the
* routines calcImgProj() and calcImgProjJacXXX() which
* compute the predicted projection & jacobian of a SINGLE 3D point (see imgproj.cpp).
* In the terminology of TR-340, these routines compute Q and its jacobians A=dQ/da, B=dQ/db.
* Notice also that what follows is two pairs of "func" and corresponding "fjac" routines.
* The first is to be used in full (i.e. motion + structure) BA, the second in
* motion only BA.
*/

static const double zerorotquat[FULLQUATSZ] = { 1.0, 0.0, 0.0, 0.0 };

/*** MEASUREMENT VECTOR AND JACOBIAN COMPUTATION FOR THE EXPERT DRIVERS ***/

/* BUNDLE ADJUSTMENT FOR CAMERA PARAMETERS ONLY */

/* Given a parameter vector p made up of the parameters of m cameras, compute in
* hx the prediction of the measurements, i.e. the projections of 3D points in the m images.
* The measurements are returned in the order (hx_11^T, .. hx_1m^T, ..., hx_n1^T, .. hx_nm^T)^T,
* where hx_ij is the predicted projection of the i-th point on the j-th camera.
* Notice that depending on idxij, some of the hx_ij might be missing
*
*/
static void img_projsRT_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata)
{
	register int i, j;
	int cnp, pnp, mnp;
	double *pqr, *pt, *ppt, *pmeas, *Kparms, *ptparams, *pr0, lrot[FULLQUATSZ], trot[FULLQUATSZ];
	//int n;
	int m, nnz;
	struct globs_ *gl;

	gl = (struct globs_ *)adata;
	cnp = gl->cnp; pnp = gl->pnp; mnp = gl->mnp;
	Kparms = gl->intrcalib;
	ptparams = gl->ptparams;

	//n=idxij->nr;
	m = idxij->nc;

	for (j = 0; j<m; ++j){
		/* j-th camera parameters */
		pqr = p + j*cnp;
		pt = pqr + 3; // quaternion vector part has 3 elements
		pr0 = gl->rot0params + j*FULLQUATSZ; // full quat for initial rotation estimate
		_MK_QUAT_FRM_VEC(lrot, pqr);
		quatMultFast(lrot, pr0, trot); // trot=lrot*pr0

		nnz = sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

		for (i = 0; i<nnz; ++i){
			ppt = ptparams + rcsubs[i] * pnp;
			pmeas = hx + idxij->val[rcidxs[i]] * mnp; // set pmeas to point to hx_ij

			calcImgProjFullR(Kparms, trot, pt, ppt, pmeas); // evaluate Q in pmeas
			//calcImgProj(Kparms, pr0, pqr, pt, ppt, pmeas); // evaluate Q in pmeas
		}
	}
}

/* Given a parameter vector p made up of the parameters of m cameras, compute in jac
* the jacobian of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
* The jacobian is returned in the order (A_11, ..., A_1m, ..., A_n1, ..., A_nm),
* where A_ij=dx_ij/db_j (see HZ).
* Notice that depending on idxij, some of the A_ij might be missing
*
*/
static void img_projsRT_jac_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata)
{
	register int i, j;
	int cnp, pnp, mnp;
	double *pqr, *pt, *ppt, *pA, *Kparms, *ptparams, *pr0;
	//int n;
	int m, nnz, Asz;
	struct globs_ *gl;

	gl = (struct globs_ *)adata;
	cnp = gl->cnp; pnp = gl->pnp; mnp = gl->mnp;
	Kparms = gl->intrcalib;
	ptparams = gl->ptparams;

	//n=idxij->nr;
	m = idxij->nc;
	Asz = mnp*cnp;

	for (j = 0; j<m; ++j){
		/* j-th camera parameters */
		pqr = p + j*cnp;
		pt = pqr + 3; // quaternion vector part has 3 elements
		pr0 = gl->rot0params + j*FULLQUATSZ; // full quat for initial rotation estimate

		nnz = sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

		for (i = 0; i<nnz; ++i){
			ppt = ptparams + rcsubs[i] * pnp;
			pA = jac + idxij->val[rcidxs[i]] * Asz; // set pA to point to A_ij

			calcImgProjJacRT(Kparms, pr0, pqr, pt, ppt, (double(*)[6])pA); // evaluate dQ/da in pA
		}
	}
}

/* BUNDLE ADJUSTMENT FOR CAMERA PARAMETERS ONLY */

/* Given a parameter vector p made up of the parameters of m cameras, compute in
* hx the prediction of the measurements, i.e. the projections of 3D points in the m images.
* The measurements are returned in the order (hx_11^T, .. hx_1m^T, ..., hx_n1^T, .. hx_nm^T)^T,
* where hx_ij is the predicted projection of the i-th point on the j-th camera.
* Notice that depending on idxij, some of the hx_ij might be missing
*
*/
static void img_projsKDRT_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata)
{
	register int i, j;
	int cnp, pnp, mnp;
	double *pqr, *pt, *ppt, *pmeas, *pcalib, *pdist, *ptparams, *pr0, lrot[FULLQUATSZ], trot[FULLQUATSZ];
	//int n;
	int m, nnz;
	struct globs_ *gl;

	gl = (struct globs_ *)adata;
	cnp = gl->cnp; pnp = gl->pnp; mnp = gl->mnp;
	ptparams = gl->ptparams;

	//n=idxij->nr;
	m = idxij->nc;

	for (j = 0; j<m; ++j){
		/* j-th camera parameters */
		pcalib = p + j*cnp;
		pdist = pcalib + 5;
		pqr = pdist + 5;
		pt = pqr + 3; // quaternion vector part has 3 elements
		pr0 = gl->rot0params + j*FULLQUATSZ; // full quat for initial rotation estimate
		_MK_QUAT_FRM_VEC(lrot, pqr);
		quatMultFast(lrot, pr0, trot); // trot=lrot*pr0

		nnz = sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

		for (i = 0; i<nnz; ++i){
			ppt = ptparams + rcsubs[i] * pnp;
			pmeas = hx + idxij->val[rcidxs[i]] * mnp; // set pmeas to point to hx_ij

			calcDistImgProjFullR(pcalib, pdist, trot, pt, ppt, pmeas); // evaluate Q in pmeas
			//calcDistImgProj(pcalib, pdist, pr0, pqr, pt, ppt, pmeas); // evaluate Q in pmeas
		}
	}
}

/* Given a parameter vector p made up of the parameters of m cameras, compute in jac
* the jacobian of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
* The jacobian is returned in the order (A_11, ..., A_1m, ..., A_n1, ..., A_nm),
* where A_ij=dx_ij/db_j (see HZ).
* Notice that depending on idxij, some of the A_ij might be missing
*
*/
static void img_projsKDRT_jac_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata)
{
	register int i, j, ii, jj;
	int cnp, pnp, mnp, ncK, ncD;
	double *pqr, *pt, *ppt, *pA, *ptr, *pcalib, *pdist, *ptparams, *pr0;
	//int n;
	int m, nnz, Asz;
	struct globs_ *gl;

	gl = (struct globs_ *)adata;
	cnp = gl->cnp; pnp = gl->pnp; mnp = gl->mnp;
	ncK = gl->nccalib;
	ncD = gl->ncdist;
	ptparams = gl->ptparams;

	//n=idxij->nr;
	m = idxij->nc;
	Asz = mnp*cnp;

	for (j = 0; j<m; ++j){
		/* j-th camera parameters */
		pcalib = p + j*cnp;
		pdist = pcalib + 5;
		pqr = pdist + 5;
		pt = pqr + 3; // quaternion vector part has 3 elements
		pr0 = gl->rot0params + j*FULLQUATSZ; // full quat for initial rotation estimate

		nnz = sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

		for (i = 0; i<nnz; ++i){
			ppt = ptparams + rcsubs[i] * pnp;
			pA = jac + idxij->val[rcidxs[i]] * Asz; // set pA to point to A_ij

			calcDistImgProjJacKDRT(pcalib, pdist, pr0, pqr, pt, ppt, (double(*)[5 + 5 + 6])pA); // evaluate dQ/da in pA

			/* clear the columns of the Jacobian corresponding to fixed calibration parameters */
			if (ncK){
				int jj0;

				ptr = pA;
				jj0 = 5 - ncK;
				for (ii = 0; ii<mnp; ++ii, ptr += cnp)
				for (jj = jj0; jj<5; ++jj)
					ptr[jj] = 0.0; // ptr[ii*cnp+jj]=0.0;
			}

			/* clear the columns of the Jacobian corresponding to fixed distortion parameters */
			if (ncD){
				int jj0;

				ptr = pA;
				jj0 = 5 - ncD;
				for (ii = 0; ii<mnp; ++ii, ptr += cnp)
				for (jj = jj0; jj<5; ++jj)
					ptr[5 + jj] = 0.0; // ptr[ii*cnp+5+jj]=0.0;
			}
		}
	}
}

/* BUNDLE ADJUSTMENT FOR CAMERA PARAMETERS ONLY */

/* Given a parameter vector p made up of the parameters of m cameras, compute in
* hx the prediction of the measurements, i.e. the projections of 3D points in the m images.
* The measurements are returned in the order (hx_11^T, .. hx_1m^T, ..., hx_n1^T, .. hx_nm^T)^T,
* where hx_ij is the predicted projection of the i-th point on the j-th camera.
* Notice that depending on idxij, some of the hx_ij might be missing
*
*/
static void img_projsKRT_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata)
{
	register int i, j;
	int cnp, pnp, mnp;
	double *pqr, *pt, *ppt, *pmeas, *pcalib, *ptparams, *pr0, lrot[FULLQUATSZ], trot[FULLQUATSZ];
	//int n;
	int m, nnz;
	struct globs_ *gl;

	gl = (struct globs_ *)adata;
	cnp = gl->cnp; pnp = gl->pnp; mnp = gl->mnp;
	ptparams = gl->ptparams;

	//n=idxij->nr;
	m = idxij->nc;

	for (j = 0; j<m; ++j){
		/* j-th camera parameters */
		pcalib = p + j*cnp;
		pqr = pcalib + 5;
		pt = pqr + 3; // quaternion vector part has 3 elements
		pr0 = gl->rot0params + j*FULLQUATSZ; // full quat for initial rotation estimate
		_MK_QUAT_FRM_VEC(lrot, pqr);
		quatMultFast(lrot, pr0, trot); // trot=lrot*pr0

		nnz = sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

		for (i = 0; i<nnz; ++i){
			ppt = ptparams + rcsubs[i] * pnp;
			pmeas = hx + idxij->val[rcidxs[i]] * mnp; // set pmeas to point to hx_ij

			calcImgProjFullR(pcalib, trot, pt, ppt, pmeas); // evaluate Q in pmeas
			//calcImgProj(pcalib, pr0, pqr, pt, ppt, pmeas); // evaluate Q in pmeas
		}
	}
}

/* Given a parameter vector p made up of the parameters of m cameras, compute in jac
* the jacobian of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
* The jacobian is returned in the order (A_11, ..., A_1m, ..., A_n1, ..., A_nm),
* where A_ij=dx_ij/db_j (see HZ).
* Notice that depending on idxij, some of the A_ij might be missing
*
*/
static void img_projsKRT_jac_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata)
{
	register int i, j, ii, jj;
	int cnp, pnp, mnp, ncK;
	double *pqr, *pt, *ppt, *pA, *pcalib, *ptparams, *pr0;
	//int n;
	int m, nnz, Asz;
	struct globs_ *gl;

	gl = (struct globs_ *)adata;
	cnp = gl->cnp; pnp = gl->pnp; mnp = gl->mnp;
	ncK = gl->nccalib;
	ptparams = gl->ptparams;

	//n=idxij->nr;
	m = idxij->nc;
	Asz = mnp*cnp;

	for (j = 0; j<m; ++j){
		/* j-th camera parameters */
		pcalib = p + j*cnp;
		pqr = pcalib + 5;
		pt = pqr + 3; // quaternion vector part has 3 elements
		pr0 = gl->rot0params + j*FULLQUATSZ; // full quat for initial rotation estimate

		nnz = sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

		for (i = 0; i<nnz; ++i){
			ppt = ptparams + rcsubs[i] * pnp;
			pA = jac + idxij->val[rcidxs[i]] * Asz; // set pA to point to A_ij

			calcImgProjJacKRT(pcalib, pr0, pqr, pt, ppt, (double(*)[5 + 6])pA); // evaluate dQ/da in pA

			/* clear the columns of the Jacobian corresponding to fixed calibration parameters */
			if (ncK){
				int jj0;

				jj0 = 5 - ncK;
				for (ii = 0; ii<mnp; ++ii, pA += cnp)
				for (jj = jj0; jj<5; ++jj)
					pA[jj] = 0.0; // pA[ii*cnp+jj]=0.0;
			}
		}
	}
}


/* BUNDLE ADJUSTMENT FOR CAMERA PARAMETERS ONLY */

/* Given the parameter vector aj of camera j, computes in xij the
* predicted projection of point i on image j
*/
static void img_projRT(int j, int i, double *aj, double *xij, void *adata)
{
	int pnp;

	double *Kparms, *pr0, *ptparams;
	struct globs_ *gl;

	gl = (struct globs_ *)adata;
	pnp = gl->pnp;
	Kparms = gl->intrcalib;
	pr0 = gl->rot0params + j*FULLQUATSZ; // full quat for initial rotation estimate
	ptparams = gl->ptparams;

	calcImgProj(Kparms, pr0, aj, aj + 3, ptparams + i*pnp, xij); // 3 is the quaternion's vector part length
}

/* Given the parameter vector aj of camera j, computes in Aij
* the jacobian of the predicted projection of point i on image j
*/
static void img_projRT_jac(int j, int i, double *aj, double *Aij, void *adata)
{
	int pnp;

	double *Kparms, *ptparams, *pr0;
	struct globs_ *gl;

	gl = (struct globs_ *)adata;
	pnp = gl->pnp;
	Kparms = gl->intrcalib;
	pr0 = gl->rot0params + j*FULLQUATSZ; // full quat for initial rotation estimate
	ptparams = gl->ptparams;

	calcImgProjJacRT(Kparms, pr0, aj, aj + 3, ptparams + i*pnp, (double(*)[6])Aij); // 3 is the quaternion's vector part length
}

/* BUNDLE ADJUSTMENT FOR STRUCTURE PARAMETERS ONLY */

/* Given the parameter vector bi of point i, computes in xij the
* predicted projection of point i on image j
*/
static void img_projS(int j, int i, double *bi, double *xij, void *adata)
{
	int cnp;

	double *Kparms, *camparams, *aj;
	struct globs_ *gl;

	gl = (struct globs_ *)adata;
	cnp = gl->cnp;
	Kparms = gl->intrcalib;
	camparams = gl->camparams;
	aj = camparams + j*cnp;

	calcImgProjFullR(Kparms, aj, aj + 3, bi, xij); // 3 is the quaternion's vector part length
	//calcImgProj(Kparms, (double *)zerorotquat, aj, aj+3, bi, xij); // 3 is the quaternion's vector part length
}

/* Given the parameter vector bi of point i, computes in Bij
* the jacobian of the predicted projection of point i on image j
*/
static void img_projS_jac(int j, int i, double *bi, double *Bij, void *adata)
{
	int cnp;

	double *Kparms, *camparams, *aj;
	struct globs_ *gl;

	gl = (struct globs_ *)adata;
	cnp = gl->cnp;
	Kparms = gl->intrcalib;
	camparams = gl->camparams;
	aj = camparams + j*cnp;

	calcImgProjJacS(Kparms, (double *)zerorotquat, aj, aj + 3, bi, (double(*)[3])Bij); // 3 is the quaternion's vector part length
}


/* BUNDLE ADJUSTMENT FOR CAMERA PARAMETERS ONLY */

/* Given the parameter vector aj of camera j, computes in xij the
* predicted projection of point i on image j
*/
static void img_projKDRT(int j, int i, double *aj, double *xij, void *adata)
{
	int pnp;

	double *ptparams, *pr0;
	struct globs_ *gl;

	gl = (struct globs_ *)adata;
	pnp = gl->pnp;
	ptparams = gl->ptparams;
	pr0 = gl->rot0params + j*FULLQUATSZ; // full quat for initial rotation estimate

	calcDistImgProj(aj, aj + 5, pr0, aj + 5 + 5, aj + 5 + 5 + 3, ptparams + i*pnp, xij); // 5 for the calibration + 5 for the distortion + 3 for the quaternion's vector part
}

/* Given the parameter vector aj of camera j, computes in Aij
* the jacobian of the predicted projection of point i on image j
*/
static void img_projKDRT_jac(int j, int i, double *aj, double *Aij, void *adata)
{
	struct globs_ *gl;
	double *pA, *ptparams, *pr0;
	int pnp, nc;

	gl = (struct globs_ *)adata;
	pnp = gl->pnp;
	ptparams = gl->ptparams;
	pr0 = gl->rot0params + j*FULLQUATSZ; // full quat for initial rotation estimate

	calcDistImgProjJacKDRT(aj, aj + 5, pr0, aj + 5 + 5, aj + 5 + 5 + 3, ptparams + i*pnp, (double(*)[5 + 5 + 6])Aij); // 5 for the calibration + 5 for the distortion + 3 for the quaternion's vector part

	/* clear the columns of the Jacobian corresponding to fixed calibration parameters */
	nc = gl->nccalib;
	if (nc){
		int cnp, mnp, j0;

		pA = Aij;
		cnp = gl->cnp;
		mnp = gl->mnp;
		j0 = 5 - nc;

		for (i = 0; i<mnp; ++i, pA += cnp)
		for (j = j0; j<5; ++j)
			pA[j] = 0.0; // pA[i*cnp+j]=0.0;
	}
	nc = gl->ncdist;
	if (nc){
		int cnp, mnp, j0;

		pA = Aij;
		cnp = gl->cnp;
		mnp = gl->mnp;
		j0 = 5 - nc;

		for (i = 0; i<mnp; ++i, pA += cnp)
		for (j = j0; j<5; ++j)
			pA[5 + j] = 0.0; // pA[i*cnp+5+j]=0.0;
	}
}

/* BUNDLE ADJUSTMENT FOR CAMERA PARAMETERS ONLY */

/* Given the parameter vector aj of camera j, computes in xij the
* predicted projection of point i on image j
*/
static void img_projKRT(int j, int i, double *aj, double *xij, void *adata)
{
	int pnp;

	double *ptparams, *pr0;
	struct globs_ *gl;

	gl = (struct globs_ *)adata;
	pnp = gl->pnp;
	ptparams = gl->ptparams;
	pr0 = gl->rot0params + j*FULLQUATSZ; // full quat for initial rotation estimate

	calcImgProj(aj, pr0, aj + 5, aj + 5 + 3, ptparams + i*pnp, xij); // 5 for the calibration + 3 for the quaternion's vector part
}

/* Given the parameter vector aj of camera j, computes in Aij
* the jacobian of the predicted projection of point i on image j
*/
static void img_projKRT_jac(int j, int i, double *aj, double *Aij, void *adata)
{
	struct globs_ *gl;
	double *ptparams, *pr0;
	int pnp, ncK;

	gl = (struct globs_ *)adata;
	pnp = gl->pnp;
	ptparams = gl->ptparams;
	pr0 = gl->rot0params + j*FULLQUATSZ; // full quat for initial rotation estimate

	calcImgProjJacKRT(aj, pr0, aj + 5, aj + 5 + 3, ptparams + i*pnp, (double(*)[5 + 6])Aij); // 5 for the calibration + 3 for the quaternion's vector part

	/* clear the columns of the Jacobian corresponding to fixed calibration parameters */
	ncK = gl->nccalib;
	if (ncK){
		int cnp, mnp, j0;

		cnp = gl->cnp;
		mnp = gl->mnp;
		j0 = 5 - ncK;

		for (i = 0; i<mnp; ++i, Aij += cnp)
		for (j = j0; j<5; ++j)
			Aij[j] = 0.0; // Aij[i*cnp+j]=0.0;
	}
}


//para_camera is always 11 double, cnp indicate if fix K
//assume para_camera: 5 for intrinsics and 3 for rotation and 3 for translation
bool SbaMotionOnly(sba_warper_data& sba)
{
	int expert = 1, verbose = 1;
	int analyticjac = 1;	/* analytic or approximate jacobian? */
	int havedist = 0;

	double opts[SBA_OPTSSZ], info[SBA_INFOSZ];

	globs_ globs;

	/* set up globs structure */
	globs.cnp = sba.cnp; globs.pnp = sba.pnp; globs.mnp = sba.mnp;
	globs.rot0params = new double[FULLQUATSZ*sba.ncamera];
	globs.ncamera = sba.ncamera; globs.n3dpoints = sba.n3dpoints;
	globs.n2dpoints = sba.n2dpoints;
	globs.ncdist = -1;

	//estimate intrinsic parameters
	globs.intrcalib = 0;
	globs.nccalib = 0; //all parameters [fu, u0, v0, ar, skew] are free, where ar is the aspect ratio fv/fu.
	
	int fixedcal = sba.fix_K;	/* varying or fixed intrinsics */
	if (fixedcal){
		globs.intrcalib = new double[5];
		globs.intrcalib[0] = sba.K[0];
		globs.intrcalib[1] = sba.K[1];
		globs.intrcalib[2] = sba.K[2];
		globs.intrcalib[3] = sba.K[3];
		globs.intrcalib[4] = sba.K[4];
	}
	//cnp==11,  5 intrinsics and 3 for rotation(image part of quaternion) and 3 for translation
	//cnp==6, 3 for rotation and 3 for translation
	for (int i = 0; i < sba.ncamera; i++){
		globs.rot0params[i + 1] = sba.para_camera[(i + 1)*sba.cnp - 6];
		globs.rot0params[i + 2] = sba.para_camera[(i + 1)*sba.cnp - 5];
		globs.rot0params[i + 3] = sba.para_camera[(i + 1)*sba.cnp - 4];
		globs.rot0params[i + 0] = std::sqrt(1.0 
			- globs.rot0params[i + 1] * globs.rot0params[i + 1]
			- globs.rot0params[i + 2] * globs.rot0params[i + 2]
			- globs.rot0params[i + 3] * globs.rot0params[i + 3]);
		/* initialize the local rotation estimates to 0, corresponding to local quats (1, 0, 0, 0) */
		sba.para_camera[(i + 1)*sba.cnp - 6] = 0.0;
		sba.para_camera[(i + 1)*sba.cnp - 5] = 0.0;
		sba.para_camera[(i + 1)*sba.cnp - 4] = 0.0;
	}
	
	//copy the 3d points x y z
	globs.ptparams = new double[sba.n3dpoints*sba.pnp];
	memcpy(globs.ptparams, sba.para_3dpoints, sba.n3dpoints*sba.pnp*sizeof(double));
	globs.camparams = 0;
	
	/* call sparse LM routine */
	opts[0] = SBA_INIT_MU; opts[1] = SBA_STOP_THRESH; opts[2] = SBA_STOP_THRESH;
	opts[3] = SBA_STOP_THRESH;
	//opts[3]=0.05*numprojs; // uncomment to force termination if the average reprojection error drops below 0.05
	opts[4] = 0.0;
	//opts[4]=1E-05; // uncomment to force termination if the relative reduction in the RMS reprojection error drops below 1E-05

	int n = 0;

	std::ofstream os("sba_warper_debug1.txt", std::ios::out | std::ios::trunc);
	if (!os){
		global::cout << "sba_warper_debug1.txt file open fail." << std::endl;
	}
	globs.print(os);
	os.close();
	
	global::cout << "before ba camara_para: ";
	for (int i = 6; i > 0; i--){
		global::cout << sba.para_camera[ 5 + i] << " ";
	}
	global::cout << endl;

	if (expert){
		n = sba_mot_levmar_x(sba.n3dpoints, sba.ncamera, 0, sba.vmask, 
			sba.para_camera, sba.cnp, sba.para_2dpoints, 0, sba.mnp,
			fixedcal ? img_projsRT_x : (havedist ? img_projsKDRT_x : img_projsKRT_x),
			analyticjac ? (fixedcal ? img_projsRT_jac_x : (havedist ? img_projsKDRT_jac_x : img_projsKRT_jac_x)) : 0,
			(void *)(&globs), 100, verbose, opts, info);
	}
	else{
		n = sba_mot_levmar(sba.n3dpoints, sba.ncamera, 0, sba.vmask, 
			sba.para_camera, sba.cnp, sba.para_2dpoints, 0, sba.mnp,
			fixedcal ? img_projRT : (havedist ? img_projKDRT : img_projKRT),
			analyticjac ? (fixedcal ? img_projRT_jac : (havedist ? img_projKDRT_jac : img_projKRT_jac)) : 0,
			(void *)(&globs), 100, verbose, opts, info);
	}
	/*
	info[6]=reason for terminating:
	1 - stopped by small gradient J^T e
    2 - stopped by small dp
	3 - stopped by itmax
	4 - stopped by small relative reduction in ||e||_2
	5 - stopped by small ||e||_2
	6 - too many attempts to increase damping. Restart with increased mu
	7 - stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
	*/
	return (info[6] < 6);
}

/********************** struct globs_ ****************************/
void globs_::print(std::ostream& os)
{
	using std::endl;
	os << " ncamera: " << ncamera << " n3dpoints: " << n3dpoints << " n2dpoints: " << n2dpoints << endl;
	os << "rot0params: " << endl;
	for (int i = 0; rot0params && i < ncamera; i++){
		os	<< rot0params[4 * i + 0] << " "
			<< rot0params[4 * i + 1] << " "
			<< rot0params[4 * i + 2] << " "
			<< rot0params[4 * i + 3] << " "
			<< endl;
	}

	os << " nccalib: " << nccalib << " ncdist: " << ncdist;
	os << endl << " intrcalib: " << endl;
	for (int i = 0; intrcalib && i < ncamera; i++){
		os	<< intrcalib[5 * i + 0] << " "
			<< intrcalib[5 * i + 1] << " "
			<< intrcalib[5 * i + 2] << " "
			<< intrcalib[5 * i + 3] << " "
			<< intrcalib[5 * i + 4] << " "
			<< endl;
	}

	os << endl << " cnp: " << cnp << " pnp: " << pnp << " mnp: " << mnp;
	os << endl << " camparams: " << endl;
	for (int i = 0; camparams && i < ncamera; i++){
		for (int j = 0; j < cnp; j++){
			os << camparams[i*cnp + j] <<" ";
		}
		os << endl;
	}

	os << " ptparams: " << endl;
	for (int i = 0; ptparams && i < n3dpoints; i++){
		os << ptparams[i*pnp + 0] << " ";
		os << ptparams[i*pnp + 1] << " ";
		os << ptparams[i*pnp + 2] << endl;
	}
}


/******************* struct sba_warper_data ***********************/
sba_warper_data::sba_warper_data() : fix_K(0), K(0),
para_camera(0), ncamera(1), cnp(6), vmask(0),
para_3dpoints(0), n3dpoints(0), pnp(3),
para_2dpoints(0), n2dpoints(0), mnp(2)
{

}

//destructor
sba_warper_data::~sba_warper_data(){
	clear();
}

void sba_warper_data::clear()
{
	if (K)				delete[] K;
	if (para_camera)	delete[] para_camera;
	if (vmask)			delete[] vmask;
	if (para_2dpoints)	delete[] para_2dpoints;
	if (para_3dpoints)	delete[] para_3dpoints;

	K = 0;
	para_camera = 0;
	vmask = 0;
	para_2dpoints = 0;
	para_3dpoints = 0;

	fix_K = 0;
	ncamera = 0; cnp = 0;
	n3dpoints = 0; pnp = 0;
	n2dpoints = 0; mnp = 0;
}


//copy constructor
sba_warper_data::sba_warper_data(const sba_warper_data&s){
	ncamera = s.ncamera;
	cnp = s.cnp;
	n3dpoints = s.n3dpoints;
	pnp = s.pnp;
	n2dpoints = s.n2dpoints;
	mnp = s.mnp;
	if (s.para_camera){
		para_camera = new double[ncamera*11];
		std::memcpy(para_camera, s.para_camera, sizeof(double)*ncamera*11);
	}
	if (s.vmask){
		vmask = new char[ncamera*n3dpoints];
		std::memcpy(vmask, s.vmask, sizeof(char)*ncamera*n3dpoints);
	}
	if (s.para_2dpoints){
		para_2dpoints = new double[n2dpoints*mnp];
		std::memcpy(para_2dpoints, s.para_2dpoints, sizeof(double)*n2dpoints*mnp);
	}
	if (s.para_3dpoints){
		para_3dpoints = new double[n3dpoints*pnp];
		std::memcpy(para_3dpoints, s.para_3dpoints, sizeof(double)*n3dpoints*pnp);
	}
}
//copy assignment
sba_warper_data& sba_warper_data::operator = (const sba_warper_data&s)
{
	if (this != &s){
		sba_warper_data temp(s);
		swap(temp);
	}
	return *this;
}

//swap function
void sba_warper_data::swap(sba_warper_data&s)
{
	std::swap(para_camera, s.para_camera);
	std::swap(ncamera, s.ncamera);
	std::swap(cnp, s.cnp);
	std::swap(vmask, s.vmask);
	std::swap(para_3dpoints, s.para_3dpoints);
	std::swap(n3dpoints, s.n3dpoints);
	std::swap(pnp, s.pnp);
	std::swap(para_2dpoints, s.para_2dpoints);
	std::swap(n2dpoints, s.n2dpoints);
	std::swap(mnp, s.mnp);
}

// print the sba data to debug
void sba_warper_data::print(std::ostream& os)
{
	using std::endl;

	os << endl;
	os << " ncamera: " << ncamera << " cnp: " << cnp << endl;
	os << " para_camera: " << endl;
	for (int i = 0; i < ncamera; i++){
		for (int j = 0; j < cnp; j++){
			os << para_camera[cnp*i + j] << " ";
		}
		os << endl;
	}

#if 1
	os << endl;
	os << " vmask: " << endl;
	for (int i = 0; i < ncamera; i++){
		for (int j = 0; j < n3dpoints; j++){
			os << (int)vmask[n3dpoints*i + j] << " ";
		}
		os << endl;
	}

	os << endl;
	os << " n3dpoints: " << n3dpoints << " pnp: " << pnp << endl;
	os << " para_3dpoints: " << endl;
	for (int i = 0; i < n3dpoints; i++){
		for (int j = 0; j < pnp; j++){
			os << para_3dpoints[pnp*i + j] << " ";
		}
		os << endl;
	}

	os << endl;
	os << " n2dpoints: " << n2dpoints << " mnp: " << mnp << endl;
	os << " para_2dpoints: " << endl;
	for (int i = 0; i < n2dpoints; i++){
		for (int j = 0; j < mnp; j++){
			os << para_2dpoints[mnp*i + j] << " ";
		}
		os << endl;
	}
#endif
}