package com.dimotim.test;

import java.math.BigInteger;
import java.util.Arrays;
import java.util.Random;
import java.util.function.LongUnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import java.util.stream.Stream;

import static com.dimotim.test.FFT.*;


class FFT {
    static final long M = 1 + 7 * 17 * (1L << 23);
    static final long g = 31; // order 2 ^ 23
    static final long log2OrdG = 23;

    static long getPrimitive(long order) {
        long log = log2(order);
        return pow2n(g, log2OrdG - log);
    }

    static long getPrimitiveInv(long order) {
        long log = log2(order);
        long gInv = invByEuclid(g);
        return pow2n(gInv, log2OrdG - log);
    }

    static long log2(long n) {
        long log2 = 0;
        while (n != 1) {
            if (n %2 != 0) throw new RuntimeException();
            n >>= 1;
            log2 ++;
        }

        return log2;
    }

    static long pow2n(long a, long log2Ord) {
        if (log2Ord == 0) {
            return a;
        } else {
            return pow2n((a * a) % M, log2Ord - 1);
        }
    }

    static long invByEuclid(long b) {
        long a = M;
        long[] ac = {1, 0};
        long[] bc = {0, 1};

        while (b != 0) {
            long q = a / b;
            long r = a % b;
            a = b;
            b = r;
            long[] ac1 = bc;
            long[] bc1 = new long[] {ac[0] - q * bc[0], ac[1] - q * bc[1]};
            ac = ac1;
            bc = bc1;
        }

        return (ac[1] + M) % M;
    }

    private static class ArrView {
        long[] ms;
        int start;
        int length;
        int step;

        public ArrView(long[] ms) {
            this.ms = ms;
            start = 0;
            length = ms.length;
            step = 1;
        }

        public ArrView(long[] ms, int start, int length, int step) {
            this.ms = ms;
            this.start = start;
            this.length = length;
            this.step = step;
        }

        int getLength() {
            return length;
        }

        long getAt(int i) {
            return ms[start + i * step];
        }

        ArrView evens() {
            return new ArrView(ms, start, length / 2, step * 2);
        }

        ArrView odds() {
            return new ArrView(ms, start + step, length / 2, step * 2);
        }
    }

    private static class ArrViewAns {
        long[] ms;
        int start;
        int length;

        public ArrViewAns(long[] ms) {
            this.ms = ms;
            start = 0;
            length = ms.length;
        }

        public ArrViewAns(long[] ms, int start, int length) {
            this.ms = ms;
            this.start = start;
            this.length = length;
        }

        long getAt(int i) {
            return ms[start + i];
        }

        void setAt(int i, long v) {
            ms[start + i] = v;
        }

        ArrViewAns left() {
            return new ArrViewAns(ms, start, length / 2);
        }

        ArrViewAns right() {
            return new ArrViewAns(ms, start + length / 2, length / 2);
        }
    }

    /**
     *
     * F(w) = A(w^2) + w * B(w^2)
     *
     */
    static void fft(ArrView fs, long prim, ArrViewAns ans) {
        int length = fs.getLength();
        if (length == 1) {
            ans.setAt(0, fs.getAt(0));
            return;
        }

        ArrView as = fs.evens();
        ArrView bs = fs.odds();

        ArrViewAns fftA = ans.left();
        ArrViewAns fftB = ans.right();

        fft(as, (prim * prim) % M, fftA);
        fft(bs, (prim * prim) % M, fftB);

        long[] prims = LongStream.iterate(1, i -> (i * prim) % M).limit(fs.length).toArray();

        int halfLength = length / 2;
        //long primN = 1;

        for (int i = 0; i < halfLength; i++) {
            long fftAi = fftA.getAt(i);
            long fftBi = fftB.getAt(i);
            long ffti = (fftAi + (prims[i] * fftBi) % M) % M;
            long fft2i = (fftAi + (prims[i + halfLength] * fftBi) % M) % M;

            ans.setAt(i, ffti);
            ans.setAt(i + halfLength, fft2i);
            //primN = (primN * prim) % M;
        }
    }

    public static long[] fft(long[] fs) {
        long[] m = new long[fs.length];
        fft(new ArrView(fs), getPrimitive(fs.length), new ArrViewAns(m));
        return m;
    }

    public static long[] fftInv(long[] fs) {
        long invN = invByEuclid(fs.length);
        long[] m = new long[fs.length];
        fft(new ArrView(fs), getPrimitiveInv(fs.length), new ArrViewAns(m));
        return Arrays.stream(m)
                .map(a -> (invN * a) % M)
                .toArray();
    }

    public static long[] polyMul(long[] as, long[] bs) {
        int maxSize = Math.max(as.length, bs.length);
        int n = IntStream.iterate(1, i -> i * 2).filter(i -> i >= maxSize).findFirst().getAsInt() * 2;
        long[] asPadded = LongStream.concat(Arrays.stream(as), LongStream.generate(() -> 0)).limit(n).toArray();
        long[] bsPadded = LongStream.concat(Arrays.stream(bs), LongStream.generate(() -> 0)).limit(n).toArray();

        long[] asFFT = fft(asPadded);
        long[] bsFFT = fft(bsPadded);
        long[] rFFT = IntStream.range(0, n).mapToLong(i -> (asFFT[i] * bsFFT[i]) % M).toArray();

        return fftInv(rFFT);
    }

    public static long[] numberMul(long[] a, long[] b) {
        long[] ch = polyMul(a, b);
        for (int i = 0; i < ch.length - 1; i++) {
            ch[i + 1] += ch[i] / BASE;
            ch[i] = ch[i] % BASE;
        }
        int l = IntStream.iterate(ch.length - 1, i -> i >= 0,  i -> i - 1).filter(i -> ch[i] != 0).findFirst().orElse(ch.length - 1);

        return Arrays.stream(ch).limit(l + 1).toArray();
    }

    static final int BASE = 100;
    static final int LOG10_BASE = 2;

    static String addPadding(String s) {
        return Stream.generate(() -> "0").limit((LOG10_BASE - s.length() % LOG10_BASE) % LOG10_BASE).collect(Collectors.joining()) + s;
    }

    static long[] numberFromString(String a) {
        String aPadded = addPadding(a);
        return IntStream.range(0, aPadded.length() / LOG10_BASE)
                .mapToLong(i -> Long.parseLong("1" + aPadded.substring(aPadded.length() - i * LOG10_BASE - LOG10_BASE, aPadded.length() - i * LOG10_BASE)) - BASE)
                .toArray();
    }

    static String numberToString(long[] a) {
        String s = IntStream.range(0, a.length)
                .mapToObj(i -> String.valueOf(a[a.length - 1 - i] + BASE).substring(1))
                .collect(Collectors.joining());

        int p = IntStream.range(0, s.length())
                .filter(i -> s.charAt(i) != '0')
                .findFirst()
                .orElse(s.length() - 1);

        return s.substring(p);
    }

    public static String numberMul(String a, String b) {
        long[] al = numberFromString(a);
        long[] bl = numberFromString(b);
        long[] r = numberMul(al, bl);
        return numberToString(r);
    }

}

class Test {

    static long ord(long e) {
        long i = 1;
        long ek = e;

        while (ek != 1) {
            ek = (ek * e) % M;
            i++;
        }

        return i;
    }

    public static long[] fftNaive(long[] fs, long prim) {
        LongUnaryOperator evalP = x -> Arrays.stream(fs).reduce(0, (a, c) -> ((a * x) % M + c) % M);

        return LongStream.iterate(1, i -> (i * prim) % M)
                .limit(fs.length)
                .map(evalP)
                .toArray();
    }

    public static void main(String[] args) {
        /*
        System.out.println(ord(g));
        System.out.println(ord(g * g));
        System.out.println(ord(g * g * g));
        System.out.println(1 << 23);


        System.out.println(ord(getPrimitive(2)));
        System.out.println((getPrimitive(2) * getPrimitive(2)) % M);

        System.out.println(ord(getPrimitive(4)));
        System.out.println(ord(getPrimitive(8)));
        System.out.println(ord(getPrimitive(16)));
        System.out.println(ord(getPrimitive(32)));

        System.out.println(ord(getPrimitive(1 << 23)));
        System.out.println(1<<23);


        System.out.println(Arrays.toString(fftNaive(new long[]{1, 2, 3, 4}, getPrimitive(4))));
        System.out.println(Arrays.toString(fft(new long[]{1, 2, 3, 4}, getPrimitive(4))));

        System.out.println(Arrays.toString(fftNaive(new long[]{1, 2, 3, 4}, getPrimitive(4))));
        System.out.println(Arrays.toString(fft(new long[]{1, 2, 3, 4})));

        System.out.println(Arrays.toString(fftNaive(fftNaive(new long[]{1, 2, 3, 4}, getPrimitive(4)), getPrimitiveInv(4))));

        System.out.println(Arrays.toString(fft(new long[]{1, 2, 3, 4})));
        System.out.println(Arrays.toString(fftInv(fft(new long[]{1, 2, 3, 4}))));
        System.out.println(M);

        System.out.println((pow2n(g, 22) * g) % M);

        System.out.println(invByEuclid(g));
        System.out.println((invByEuclid(g) * g) % M);
        System.out.println((invByEuclid(4) * 4) % M);
        System.out.println((invByEuclid(getPrimitive(8)) * getPrimitive(8)) % M);

        System.out.println(ord(getPrimitiveInv(8)));

        System.out.println(Arrays.toString(fftInv(fft(new long[]{11, 22, 33, 44}))));


        System.out.println(Arrays.toString(polyMul(new long[]{1,4,6,4,1}, new long[]{1,4,6,4,1})));
        System.out.println(Arrays.toString(polyMul(new long[]{3,2,1}, new long[]{1,4})));


        System.out.println(Arrays.toString(numberMul(new long[]{999, 999}, new long[]{999, 999})));
        */

        Random random = new Random();
        int n = 250000;
        //int n = 20;
        String a = "9" + IntStream.range(0, n).mapToObj(r -> random.nextInt(10)).map(String::valueOf).collect(Collectors.joining());
        String b = "6" + IntStream.range(0, n).mapToObj(r -> random.nextInt(10)).map(String::valueOf).collect(Collectors.joining());


        long t1 = System.currentTimeMillis();
        String r1 = numberMul(a, b);
        System.out.println(System.currentTimeMillis() - t1);
        //System.out.println(r1);

        long t2 = System.currentTimeMillis();
        String r2 = new BigInteger(a).multiply(new BigInteger(b)).toString();
        System.out.println(System.currentTimeMillis() - t2);
        //System.out.println(r2);

        System.out.println(r1.equals(r2));
    }
}