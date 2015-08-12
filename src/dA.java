/**
 * Created by bit on 2015-07-22.
 */
import java.util.Random;

public class dA {
    public int N;


    public int n_visible;
    public int n_hidden;
    public double[][] W;
    public double[] hbias;
    public double[] vbias;
    public Random rng;


    public double uniform(double min, double max) {
        return rng.nextDouble() * (max - min) + min;
    }

    public int binomial(int n, double p) {
        if(p < 0 || p > 1) return 0;

        int c = 0;
        double r;

        for(int i=0 ; i < n ; i++) {
            r = rng.nextDouble();
            if (r < p) c++;
        }

        return c;
    } // End of binomial

    public static double sigmoid(double x) {

        return 1.0 / ( 1.0 + Math.pow(Math.E, -x) );   // 1 / (1+e^-x) ;
    } //End of sigmoid function

    public dA(int N, int n_visible, int n_hidden , double[][] W, double[] hbias, double[] vbias, Random rng) {
        // Initialization of Weights(W) , Hidden bias(hbias), Visible bias(vbias)

        this.N = N;
        this.n_visible = n_visible;
        this.n_hidden = n_hidden;

        if(rng == null)	this.rng = new Random(1234);
        else this.rng = rng;

        //Initialization of Weights
        if( W == null ) {
            this.W = new double[this.n_hidden][this.n_visible];
            double a = 1.0 / this.n_visible;

            for(int i=0; i<this.n_hidden ; i++) {
                for(int j=0; j<this.n_visible ; j++) {
                    this.W[i][j] = uniform(-a, a);
                }
            }
        } else {
            this.W = W;
        }

        //Initialization of hbias
        if(hbias == null) {
            this.hbias = new double[this.n_hidden];
            for(int i=0; i<this.n_hidden; i++)
                this.hbias[i] = 0;
        } else {
            this.hbias = hbias;
        }

        //Initialization of vbias
        if(vbias == null) {
            this.vbias = new double[this.n_visible];
            for(int i=0; i<this.n_visible; i++)
                this.vbias[i] = 0;
        } else {
            this.vbias = vbias;
        }

    } //end of dA method

    public void get_corrupted_input(int[] x, int[] tilde_x, double p) {
        for(int i=0; i < n_visible; i++) {
            if(x[i] == 0) {
                tilde_x[i] = 0;
            } else {
                tilde_x[i] = binomial(1, p);
            }
        }
    } //End of get_corrupted_input

    // Encode
    public void get_hidden_values(int[] x, double[] y) {
        for(int i=0; i < n_hidden; i++) {
            y[i] = 0;
            for(int j=0; j < n_visible; j++) {
                y[i] += W[i][j] * x[j];
            }
            y[i] += hbias[i];
            y[i] = sigmoid(y[i]);
        }
    } //End od get_hidden_values

    // Decode
    public void get_reconstructed_input(double[] y, double[] z) {
        for(int i=0; i<n_visible; i++) {
            z[i] = 0;
            for(int j=0; j<n_hidden; j++) {
                z[i] += W[j][i] * y[j];
            }
            z[i] += vbias[i];
            z[i] = sigmoid(z[i]);
        }
    } //End of get_reconstructed_input method

    //Training Step
    public void train(int[] x, double lr, double corruption_level) {
        // x[] : input layer , lr : learning_rate , corruption_level : ?
        int[] tilde_x = new int[n_visible];
        double[] y = new double[n_hidden];
        double[] z = new double[n_visible];

        double[] L_vbias = new double[n_visible]; //Bias of visible layer
        double[] L_hbias = new double[n_hidden];  //Bias of hidden layer

        double p = 1 - corruption_level; // What do u mean by that?

        get_corrupted_input(x, tilde_x, p); // tilde_x = value of binomial distribution
        get_hidden_values(tilde_x, y);  //Encode
        get_reconstructed_input(y, z);  //Decode

        // Bias of visible layer && Initialization of that.
        for( int i=0 ; i < n_visible ; i++ ) {
            L_vbias[i] = ( x[i] - z[i] ); //   Input layer - output layer , round(L)/round(h2)
            vbias[i] += lr * L_vbias[i] / N;  // vbias -> round(L)/round(b`) where h2(y) = W`y+b`  (17)
        }

        // Bias of hidden layer && Initialization of that
        for( int i=0 ; i < n_hidden ; i++ ) {
            L_hbias[i] = 0;
            for(int j=0 ; j < n_visible ; j++) {
                L_hbias[i] += W[i][j] * L_vbias[j]; // W(x-z)
            }
            L_hbias[i] *= y[i] * (1 - y[i]); // y[]: hidden neuron , W(x-z)*y*(1-y)
            hbias[i] += lr * L_hbias[i] / N;  //  ( 16 )   hbias ->  round(L)/round(b)  where h1(y) = Wx+ b
        }

        // Updating weights with L_hbias and L_vbias while training
        for(int i=0; i < n_hidden ; i++) {
            for(int j=0; j < n_visible ; j++) {
                W[i][j] += lr * ( L_hbias[i] * tilde_x[j] + L_vbias[j] * y[i] ) / N;  //(15)
            }
        }

    } //End of training method

    public void reconstruct(int[] x, double[] z) {
        double[] y = new double[n_hidden];
        get_hidden_values(x, y);
        get_reconstructed_input(y, z);
    } //End of reconstruct()

    private static void test_dA() {
        Random rng = new Random(123);

        double learning_rate = 0.1;
        double corruption_level = 0.3;
        int training_epochs = 100;

        int train_N = 10;
        int test_N = 2;
        int n_visible = 20;
        int n_hidden = 5;

        int[][] train_X = {
                {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0}
        };

        dA da = new dA(train_N, n_visible, n_hidden, null, null, null, rng);
        // train_N : 10, n_visible : 20 , n_hidden : 5 , rng : random number
        // train
        for(int epoch = 0 ; epoch < training_epochs /*(100 )*/; epoch++) {
            for(int i=0 ; i < train_N ; i++) {
                da.train(train_X[i], learning_rate, corruption_level);
            }
        }

        // test data
        int[][] test_X = {
                {1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                //{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0}
                {0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0}
        };

        double[][] reconstructed_X = new double[test_N][n_visible];

        // test
        for(int i=0; i < test_N; i++) {
            da.reconstruct( test_X[i], reconstructed_X[i] );
            for(int j=0; j < n_visible; j++) {
                System.out.printf("%.5f ", reconstructed_X[i][j]);
            }
            System.out.println();
        }
    } //End of test_dA()

    public static void main(String[] args) {
        test_dA();
    }
}
