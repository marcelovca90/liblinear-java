package de.bwaldvogel.liblinear;

import static java.lang.Math.sqrt;


class L2R_L2_SvcFunction implements Function {

    protected final Problem  prob;
    protected final double[] C;
    protected final int[]    I;
    protected final double[] z;
    protected double[]       D;
    private final boolean    POLY2 = true;
    protected double         coef0;
    protected double         gamma;
    protected double         sqrt2;
    protected double         sqrt2_coef0_g;
    protected double         sqrt2_g;

    protected int            sizeI;

    public L2R_L2_SvcFunction( Problem prob, double[] C ) {
        int l = prob.l;

        this.prob = prob;

        z = new double[l];
        D = new double[l];
        I = new int[l];
        this.C = C;
        if (POLY2) {
            coef0 = prob.coef0;
            gamma = prob.gamma;
            sqrt2 = sqrt(2.0);
            sqrt2_coef0_g = sqrt2 * sqrt(coef0 * gamma);
            sqrt2_g = sqrt2 * gamma;
        }
    }

    @Override
    public double fun(double[] w) {
        int i;
        double f = 0;
        double[] y = prob.y;
        int l = prob.l;
        int w_size = get_nr_variable();

        Xv(w, z);

        for (i = 0; i < w_size; i++)
            f += w[i] * w[i];
        f /= 2.0;
        for (i = 0; i < l; i++) {
            z[i] = y[i] * z[i];
            double d = 1 - z[i];
            if (d > 0) f += C[i] * d * d;
        }

        return (f);
    }

    @Override
    public int get_nr_variable() {
        if (POLY2)
            return (prob.n + 2) * (prob.n + 1) / 2;
        else
            return prob.n;
    }

    @Override
    public void grad(double[] w, double[] g) {
        double[] y = prob.y;
        int l = prob.l;
        int w_size = get_nr_variable();

        sizeI = 0;
        for (int i = 0; i < l; i++) {
            if (z[i] < 1) {
                z[sizeI] = C[i] * y[i] * (z[i] - 1);
                I[sizeI] = i;
                sizeI++;
            }
        }
        subXTv(z, g);

        for (int i = 0; i < w_size; i++)
            g[i] = w[i] + 2 * g[i];
    }

    @Override
    public void Hv(double[] s, double[] Hs) {
        int i;
        int w_size = get_nr_variable();
        Feature[][] x = prob.x;

        for (i = 0; i < w_size; i++)
            Hs[i] = 0;
        for (i = 0; i < sizeI; i++) {
            Feature[] xi = x[I[i]];
            double xTs = SparseOperator.dot(s, xi);
            xTs = C[I[i]] * xTs;

            SparseOperator.axpy(xTs, xi, Hs);
        }
        for (i = 0; i < w_size; i++)
            Hs[i] = s[i] + 2 * Hs[i];
    }

    protected void subXTv(double[] v, double[] XTv) {
        int i;
        int w_size = get_nr_variable();
        Feature[][] x = prob.x;

        for (i = 0; i < w_size; i++)
            XTv[i] = 0;

        if (POLY2) {
            int n = prob.n;
            for (i = 0; i < sizeI; i++) {
                Feature[] xi = x[I[i]];
                Feature[] xj;
                XTv[0] += v[i] * coef0;
                for (Feature fi : xi) {
                    xj = x[i + 1];
                    double[] XTv_x = SparseOperator.subarray(XTv, (fi.getIndex() * (2 * n - fi.getIndex() + 1)) / 2);
                    for (Feature fj : xj) // quadratic
                        XTv_x[fj.getIndex()] += v[i] * (fj.getValue()) * sqrt2_g * fi.getValue();
                    XTv_x[fi.getIndex()] += v[i] * fi.getValue() * fi.getValue() * gamma;
                    XTv[fi.getIndex()] += v[i] * sqrt2_coef0_g * fi.getValue();
                }
            }
        } else {
            for (i = 0; i < sizeI; i++)
                SparseOperator.axpy(v[i], x[I[i]], XTv);
        }
    }

    protected void Xv(double[] v, double[] Xv) {
        int l = prob.l;
        Feature[][] x = prob.x;
        if (POLY2) {
            double tmp_value;
            int n = prob.n;
            for (int i = 0; i < l; i++) {
                Feature[] xi = x[i];
                Feature[] xj;
                Xv[i] = v[0] * coef0;
                for (Feature fi : xi) {
                    tmp_value = 0.0;
                    xj = x[i + 1];
                    double[] v_x = SparseOperator.subarray(v, (fi.getIndex() * (2 * n - fi.getIndex() + 1)) / 2);
                    for (Feature fj : xj) // quadratic
                        tmp_value += v_x[fj.getIndex()] * (fj.getValue());
                    tmp_value *= sqrt2_g;
                    tmp_value += v_x[fi.getIndex()] * (fi.getValue()) * gamma;
                    tmp_value += v[fi.getIndex()] * sqrt2_coef0_g;
                    Xv[i] += tmp_value * (fi.getValue());
                }
            }
        } else {
            for (int i = 0; i < l; i++)
                Xv[i] = SparseOperator.dot(v, x[i]);
        }
    }

    @Override
    public void get_diagH(double[] M) {
        int w_size = get_nr_variable();
        Feature[][] x = prob.x;

        for (int i = 0; i < w_size; i++)
            M[i] = 1;

        for (int i = 0; i < sizeI; i++) {
            int idx = I[i];
            for (Feature s : x[idx]) {
                M[s.getIndex() - 1] += s.getValue() * s.getValue() * C[idx] * 2;
            }
        }
    }
}
