package de.bwaldvogel.liblinear;

import java.util.Arrays;


class SparseOperator {

    static double nrm2_sq(Feature[] x) {
        double ret = 0;
        for (Feature feature : x) {
            ret += feature.getValue() * feature.getValue();
        }
        return (ret);
    }

    static double dot(double[] v, Feature[] x) {
        double ret = 0;
        for (Feature s : x) {
            ret += v[s.getIndex() - 1] * s.getValue();
        }
        return (ret);
    }

    static double dot2(double[] v_x, Feature[] x) {
        double ret = 0;
        for (Feature xj : x) {
            ret += v_x[xj.getIndex()] * xj.getValue();
        }
        return (ret);
    }

    static void axpy(double a, Feature[] x, double[] y) {
        for (Feature feature : x) {
            y[feature.getIndex() - 1] += a * feature.getValue();
        }
    }

    static double[] subarray(double[] v, int i) {
        return Arrays.copyOfRange(v, i, v.length - 1);
    }
}
