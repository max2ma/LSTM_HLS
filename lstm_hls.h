#pragma once
#include "hls_math.h"
#include "hls_stream.h"
template<typename T>
const T sigmoid(const T value){
	T v = expf(-value);
#pragma HLS RESOURCE variable=v core=Fexp_nodsp
	T r = T(1)/(T(1) + v);
#pragma HLS RESOURCE variable=r core=FRecip_nodsp
#pragma HLS RESOURCE variable=r core=FAddSub_nodsp
	return r;
}
template<typename T>
const T tanh(const T value){
	T v2 = expf(2*value);
#pragma HLS RESOURCE variable=v2 core=Fexp_nodsp
	T r = (v2-T(1))/(T(1) + v2);
#pragma HLS RESOURCE variable=r core=FRecip_nodsp
#pragma HLS RESOURCE variable=r core=FAddSub_nodsp
}

#if 0
template<int H, int W, int PE, typename T>
void matMulAdd(const T w[W][H], const T v[W], const T bias[H], T o[H]){
	T tmp_o[H];
//#pragma HLS ALLOCATION instances=fmul limit=PE operation
	for(int i=0;i<H;i++)
#pragma HLS PIPELINE
		tmp_o[i] = bias[i];
	for(int j=0;j<W;j++){
		T tmp_in = v[j];
		for(int i=0;i<H;i++)
#pragma HLS PIPELINE
			tmp_o[i]+=w[j][i] * tmp_in;
	}
	for(int i=0;i<H;i++)
#pragma HLS PIPELINE
		o[i] = tmp_o[i];
}
#else

template<int H, int W, int PE, typename T>
void matMulAdd(const T w[W][H], const T v[W], const T bias[H], T *o){
#pragma HLS ARRAY_PARTITION variable=w complete dim=1
#pragma HLS ARRAY_PARTITION variable=v complete dim=1
	T tmp_o[H];
//#pragma HLS ALLOCATION instances=fmul limit=PE operation
	for(int i=0;i<H;i++){
//#pragma HLS PIPELINE
		T tmp_o = bias[i];
		for(int j=0;j<W;j++)
//#pragma HLS UNROLL
			tmp_o+=w[j][i] * v[j];
		o[i] = tmp_o;
	}
}
#endif

template<int N, typename T>
void add(const T a0[N], const T a1[N], T a2[N]){
	for(int i=0;i<N;i++)
#pragma HLS PIPELINE
		a2[i] = a0[i] + a1[i] ;
}

template<int N, typename T>
void add(const T a0[N], const T a1[N], const T a2[N], T a3[N]){
	for(int i=0;i<N;i++)
#pragma HLS PIPELINE
		a3[i] = a0[i] + a1[i] + a2[i];
}

template<int INPUT_SIZE, int LSTM_SIZE, typename T>
void lstm(const T *input,
		hls::stream<T> &output,
		const T BIAS[LSTM_SIZE * 4],
		const T WI[INPUT_SIZE][LSTM_SIZE * 4],
		const T WS[LSTM_SIZE][LSTM_SIZE * 4],
		const int Samples){
#pragma HLS ARRAY_PARTITION variable=BIAS block factor=2
#pragma HLS ARRAY_PARTITION variable=WI block factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=WS block factor=2 dim=2
	static T state[LSTM_SIZE] = {0};
	static T hidden[LSTM_SIZE] = {0};
#pragma HLS ARRAY_PARTITION variable=hidden complete dim=1
	T in[INPUT_SIZE];
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
		T out[LSTM_SIZE];
//#pragma HLS ARRAY_PARTITION variable=out complete dim=1
	for(int rep=0; rep<Samples;rep++){

		int start_point = rep * INPUT_SIZE;
		for(int i = 0;i<INPUT_SIZE;i++)
#pragma HLS PIPELINE
			in[i]=input[start_point+i];
		for(int i = 0; i < LSTM_SIZE; i++) {
#pragma HLS PIPELINE II = 4
//#pragma HLS ALLOCATION instances=fmul limit=40 operation
			T g0 = BIAS[i];
			T g1 = BIAS[1 * LSTM_SIZE + i];
			T g2 = BIAS[2 * LSTM_SIZE + i];
			T g3 = BIAS[3 * LSTM_SIZE + i];
			for(int j=0;j<INPUT_SIZE;j++){
				g0 +=WI[j][i + (LSTM_SIZE * 0)] * in[j];
				g1 +=WI[j][i + (LSTM_SIZE * 1)] * in[j];
				g2 +=WI[j][i + (LSTM_SIZE * 2)] * in[j];
				g3 +=WI[j][i + (LSTM_SIZE * 3)] * in[j];
			}
			for(int j=0;j<LSTM_SIZE;j++){
				g0 +=WS[j][i + (LSTM_SIZE * 0)] * hidden[j];
				g1 +=WS[j][i + (LSTM_SIZE * 1)] * hidden[j];
				g2 +=WS[j][i + (LSTM_SIZE * 2)] * hidden[j];
				g3 +=WS[j][i + (LSTM_SIZE * 3)] * hidden[j];
			}

			T s = sigmoid(g1) * state[i] + sigmoid(g0) * tanh(g2);
			state[i] = s;
			out[i] = sigmoid(g3) * tanh(s);
			output.write(out[i]);
		}
		for(int j = 0; j < LSTM_SIZE; j++)
#pragma HLS PIPELINE
			hidden[j] = out[j];
	}
}


template<int INPUT_SIZE, int OUTPUT_SIZE, typename T>
void feed_forward(hls::stream<T> &in,
		T *output,
		const T W1[INPUT_SIZE][OUTPUT_SIZE],
		const T BIAS1[OUTPUT_SIZE],
		const int Samples){
#pragma HLS INLINE off
	for(int rep=0; rep<Samples; rep++){
//#pragma HLS DATAFLOW
		T input[INPUT_SIZE];
		for(int i=0;i<INPUT_SIZE;i++)
#pragma HLS PIPELINE
			input[i] = in.read();
//		T tmp[OUTPUT_SIZE];
		matMulAdd<OUTPUT_SIZE, INPUT_SIZE, 4>(W1, input, BIAS1, output+ rep * OUTPUT_SIZE);
//		add<OUTPUT_SIZE>(tmp, BIAS1, output[rep]);
	}
}

