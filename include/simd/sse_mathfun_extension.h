/* Modified version of https://github.com/to-miz/sse_mathfun_extension for VCV Rack.

The following changes were made.
- Make functions inline.
- Change global constants to function-scope.

This derived source file is released under the zlib license.
*/

/*
sse_mathfun_extension.h - zlib license
Written by Tolga Mizrak 2016
Extension of sse_mathfun.h, which is written by Julien Pommier

Based on the corresponding algorithms of the cephes math library

This is written as an extension to sse_mathfun.h instead of modifying it, just because I didn't want
to maintain a modified version of the original library. This way switching to a newer version of the
library won't be a hassle.

Note that non SSE2 implementations of tan_ps, atan_ps, cot_ps and atan2_ps are not implemented yet.
As such, currently you need to #define USE_SSE2 to compile.

With tan_ps, cot_ps you get good precision on input ranges that are further away from the domain
borders (-PI/2, PI/2 for tan and 0, 1 for cot). See the results on the deviations for these
functions on my machine:
checking tan on [-0.25*Pi, 0.25*Pi]
max deviation from tanf(x): 1.19209e-07 at 0.250000006957*Pi, max deviation from cephes_tan(x):
5.96046e-08
   ->> precision OK for the tan_ps <<-

checking tan on [-0.49*Pi, 0.49*Pi]
max deviation from tanf(x): 3.8147e-06 at -0.490000009841*Pi, max deviation from cephes_tan(x):
9.53674e-07
   ->> precision OK for the tan_ps <<-

checking cot on [0.2*Pi, 0.7*Pi]
max deviation from cotf(x): 1.19209e-07 at 0.204303119606*Pi, max deviation from cephes_cot(x):
1.19209e-07
   ->> precision OK for the cot_ps <<-

checking cot on [0.01*Pi, 0.99*Pi]
max deviation from cotf(x): 3.8147e-06 at 0.987876517942*Pi, max deviation from cephes_cot(x):
9.53674e-07
   ->> precision OK for the cot_ps <<-

With atan_ps and atan2_ps you get pretty good precision, atan_ps max deviation is < 2e-7 and
atan2_ps max deviation is < 2.5e-7
*/

/* Copyright (C) 2016 Tolga Mizrak

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
	 claim that you wrote the original software. If you use this software
	 in a product, an acknowledgment in the product documentation would be
	 appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
	 misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/
#pragma once
#include "sse_mathfun.h"


inline simde__m128 sse_mathfun_tancot_ps(simde__m128 x, int cotFlag) {
	simde__m128 p0 = simde_mm_set_ps1(9.38540185543E-3);
	simde__m128 p1 = simde_mm_set_ps1(3.11992232697E-3);
	simde__m128 p2 = simde_mm_set_ps1(2.44301354525E-2);
	simde__m128 p3 = simde_mm_set_ps1(5.34112807005E-2);
	simde__m128 p4 = simde_mm_set_ps1(1.33387994085E-1);
	simde__m128 p5 = simde_mm_set_ps1(3.33331568548E-1);

	simde__m128 xmm1, xmm2 = simde_mm_setzero_ps(), xmm3, sign_bit, y;

	simde__m128i emm2;
	sign_bit = x;
	/* take the absolute value */
	simde__m128 sign_mask = simde_mm_castsi128_ps(simde_mm_set1_epi32(0x80000000));
	simde__m128 inv_sign_mask = simde_mm_castsi128_ps(simde_mm_set1_epi32(~0x80000000));
	x = simde_mm_and_ps(x, inv_sign_mask);
	/* extract the sign bit (upper one) */
	sign_bit = simde_mm_and_ps(sign_bit, sign_mask);

	/* scale by 4/Pi */
	simde__m128 cephes_FOPI = simde_mm_set_ps1(1.27323954473516);
	y = simde_mm_mul_ps(x, cephes_FOPI);

	/* store the integer part of y in mm0 */
	emm2 = simde_mm_cvttps_epi32(y);
	/* j=(j+1) & (~1) (see the cephes sources) */
	emm2 = simde_mm_add_epi32(emm2, simde_mm_set1_epi32(1));
	emm2 = simde_mm_and_si128(emm2, simde_mm_set1_epi32(~1));
	y = simde_mm_cvtepi32_ps(emm2);

	emm2 = simde_mm_and_si128(emm2, simde_mm_set1_epi32(2));
	emm2 = simde_mm_cmpeq_epi32(emm2, simde_mm_setzero_si128());

	simde__m128 poly_mask = simde_mm_castsi128_ps(emm2);
	/* The magic pass: "Extended precision modular arithmetic"
	   x = ((x - y * DP1) - y * DP2) - y * DP3; */
	simde__m128 minus_cephes_DP1 = simde_mm_set_ps1(-0.78515625);
	simde__m128 minus_cephes_DP2 = simde_mm_set_ps1(-2.4187564849853515625e-4);
	simde__m128 minus_cephes_DP3 = simde_mm_set_ps1(-3.77489497744594108e-8);
	xmm1 = minus_cephes_DP1;
	xmm2 = minus_cephes_DP2;
	xmm3 = minus_cephes_DP3;
	xmm1 = simde_mm_mul_ps(y, xmm1);
	xmm2 = simde_mm_mul_ps(y, xmm2);
	xmm3 = simde_mm_mul_ps(y, xmm3);
	simde__m128 z = simde_mm_add_ps(x, xmm1);
	z = simde_mm_add_ps(z, xmm2);
	z = simde_mm_add_ps(z, xmm3);

	simde__m128 zz = simde_mm_mul_ps(z, z);

	y = p0;
	y = simde_mm_mul_ps(y, zz);
	y = simde_mm_add_ps(y, p1);
	y = simde_mm_mul_ps(y, zz);
	y = simde_mm_add_ps(y, p2);
	y = simde_mm_mul_ps(y, zz);
	y = simde_mm_add_ps(y, p3);
	y = simde_mm_mul_ps(y, zz);
	y = simde_mm_add_ps(y, p4);
	y = simde_mm_mul_ps(y, zz);
	y = simde_mm_add_ps(y, p5);
	y = simde_mm_mul_ps(y, zz);
	y = simde_mm_mul_ps(y, z);
	y = simde_mm_add_ps(y, z);

	simde__m128 y2;
	if (cotFlag) {
		y2 = simde_mm_xor_ps(y, sign_mask);
		/* y = simde_mm_rcp_ps(y); */
		/* using simde_mm_rcp_ps here loses on way too much precision, better to do a div */
		y = simde_mm_div_ps(simde_mm_set_ps1(1.f), y);
	}
	else {
		/* y2 = simde_mm_rcp_ps(y); */
		/* using simde_mm_rcp_ps here loses on way too much precision, better to do a div */
		y2 = simde_mm_div_ps(simde_mm_set_ps1(1.f), y);
		y2 = simde_mm_xor_ps(y2, sign_mask);
	}

	/* select the correct result from the two polynoms */
	xmm3 = poly_mask;
	y = simde_mm_and_ps(xmm3, y);
	y2 = simde_mm_andnot_ps(xmm3, y2);
	y = simde_mm_or_ps(y, y2);

	/* update the sign */
	y = simde_mm_xor_ps(y, sign_bit);

	return y;
}

inline simde__m128 sse_mathfun_tan_ps(simde__m128 x) {
	return sse_mathfun_tancot_ps(x, 0);
}

inline simde__m128 sse_mathfun_cot_ps(simde__m128 x) {
	return sse_mathfun_tancot_ps(x, 1);
}


inline simde__m128 sse_mathfun_atan_ps(simde__m128 x) {
	simde__m128 sign_mask = simde_mm_castsi128_ps(simde_mm_set1_epi32(0x80000000));
	simde__m128 inv_sign_mask = simde_mm_castsi128_ps(simde_mm_set1_epi32(~0x80000000));

	simde__m128 atanrange_hi = simde_mm_set_ps1(2.414213562373095);
	simde__m128 atanrange_lo = simde_mm_set_ps1(0.4142135623730950);
	simde__m128 cephes_PIO2F = simde_mm_set_ps1(1.5707963267948966192);
	simde__m128 cephes_PIO4F = simde_mm_set_ps1(0.7853981633974483096);

	simde__m128 atancof_p0 = simde_mm_set_ps1(8.05374449538e-2);
	simde__m128 atancof_p1 = simde_mm_set_ps1(1.38776856032E-1);
	simde__m128 atancof_p2 = simde_mm_set_ps1(1.99777106478E-1);
	simde__m128 atancof_p3 = simde_mm_set_ps1(3.33329491539E-1);

	simde__m128 sign_bit, y;

	sign_bit = x;
	/* take the absolute value */
	x = simde_mm_and_ps(x, inv_sign_mask);
	/* extract the sign bit (upper one) */
	sign_bit = simde_mm_and_ps(sign_bit, sign_mask);

	/* range reduction, init x and y depending on range */
	/* x > 2.414213562373095 */
	simde__m128 cmp0 = simde_mm_cmpgt_ps(x, atanrange_hi);
	/* x > 0.4142135623730950 */
	simde__m128 cmp1 = simde_mm_cmpgt_ps(x, atanrange_lo);

	/* x > 0.4142135623730950 && !(x > 2.414213562373095) */
	simde__m128 cmp2 = simde_mm_andnot_ps(cmp0, cmp1);

	/* -(1.0/x) */
	simde__m128 y0 = simde_mm_and_ps(cmp0, cephes_PIO2F);
	simde__m128 x0 = simde_mm_div_ps(simde_mm_set_ps1(1.f), x);
	x0 = simde_mm_xor_ps(x0, sign_mask);

	simde__m128 y1 = simde_mm_and_ps(cmp2, cephes_PIO4F);
	/* (x-1.0)/(x+1.0) */
	simde__m128 x1_o = simde_mm_sub_ps(x, simde_mm_set_ps1(1.f));
	simde__m128 x1_u = simde_mm_add_ps(x, simde_mm_set_ps1(1.f));
	simde__m128 x1 = simde_mm_div_ps(x1_o, x1_u);

	simde__m128 x2 = simde_mm_and_ps(cmp2, x1);
	x0 = simde_mm_and_ps(cmp0, x0);
	x2 = simde_mm_or_ps(x2, x0);
	cmp1 = simde_mm_or_ps(cmp0, cmp2);
	x2 = simde_mm_and_ps(cmp1, x2);
	x = simde_mm_andnot_ps(cmp1, x);
	x = simde_mm_or_ps(x2, x);

	y = simde_mm_or_ps(y0, y1);

	simde__m128 zz = simde_mm_mul_ps(x, x);
	simde__m128 acc = atancof_p0;
	acc = simde_mm_mul_ps(acc, zz);
	acc = simde_mm_sub_ps(acc, atancof_p1);
	acc = simde_mm_mul_ps(acc, zz);
	acc = simde_mm_add_ps(acc, atancof_p2);
	acc = simde_mm_mul_ps(acc, zz);
	acc = simde_mm_sub_ps(acc, atancof_p3);
	acc = simde_mm_mul_ps(acc, zz);
	acc = simde_mm_mul_ps(acc, x);
	acc = simde_mm_add_ps(acc, x);
	y = simde_mm_add_ps(y, acc);

	/* update the sign */
	y = simde_mm_xor_ps(y, sign_bit);

	return y;
}

inline simde__m128 sse_mathfun_atan2_ps(simde__m128 y, simde__m128 x) {
	simde__m128 sign_mask = simde_mm_castsi128_ps(simde_mm_set1_epi32(0x80000000));
	simde__m128 x_eq_0 = simde_mm_cmpeq_ps(x, simde_mm_setzero_ps());
	simde__m128 x_gt_0 = simde_mm_cmpgt_ps(x, simde_mm_setzero_ps());
	simde__m128 x_le_0 = simde_mm_cmple_ps(x, simde_mm_setzero_ps());
	simde__m128 y_eq_0 = simde_mm_cmpeq_ps(y, simde_mm_setzero_ps());
	simde__m128 x_lt_0 = simde_mm_cmplt_ps(x, simde_mm_setzero_ps());
	simde__m128 y_lt_0 = simde_mm_cmplt_ps(y, simde_mm_setzero_ps());
	simde__m128 cephes_PIF = simde_mm_set_ps1(3.141592653589793238);
	simde__m128 cephes_PIO2F = simde_mm_set_ps1(1.5707963267948966192);

	simde__m128 zero_mask = simde_mm_and_ps(x_eq_0, y_eq_0);
	simde__m128 zero_mask_other_case = simde_mm_and_ps(y_eq_0, x_gt_0);
	zero_mask = simde_mm_or_ps(zero_mask, zero_mask_other_case);

	simde__m128 pio2_mask = simde_mm_andnot_ps(y_eq_0, x_eq_0);
	simde__m128 pio2_mask_sign = simde_mm_and_ps(y_lt_0, sign_mask);
	simde__m128 pio2_result = cephes_PIO2F;
	pio2_result = simde_mm_xor_ps(pio2_result, pio2_mask_sign);
	pio2_result = simde_mm_and_ps(pio2_mask, pio2_result);

	simde__m128 pi_mask = simde_mm_and_ps(y_eq_0, x_le_0);
	simde__m128 pi = cephes_PIF;
	simde__m128 pi_result = simde_mm_and_ps(pi_mask, pi);

	simde__m128 swap_sign_mask_offset = simde_mm_and_ps(x_lt_0, y_lt_0);
	swap_sign_mask_offset = simde_mm_and_ps(swap_sign_mask_offset, sign_mask);

	simde__m128 offset0 = simde_mm_setzero_ps();
	simde__m128 offset1 = cephes_PIF;
	offset1 = simde_mm_xor_ps(offset1, swap_sign_mask_offset);

	simde__m128 offset = simde_mm_andnot_ps(x_lt_0, offset0);
	offset = simde_mm_and_ps(x_lt_0, offset1);

	simde__m128 arg = simde_mm_div_ps(y, x);
	simde__m128 atan_result = sse_mathfun_atan_ps(arg);
	atan_result = simde_mm_add_ps(atan_result, offset);

	/* select between zero_result, pio2_result and atan_result */

	simde__m128 result = simde_mm_andnot_ps(zero_mask, pio2_result);
	atan_result = simde_mm_andnot_ps(pio2_mask, atan_result);
	atan_result = simde_mm_andnot_ps(pio2_mask, atan_result);
	result = simde_mm_or_ps(result, atan_result);
	result = simde_mm_or_ps(result, pi_result);

	return result;
}
