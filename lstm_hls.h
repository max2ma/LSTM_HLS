#pragma once
#include "hls_math.h"

template<typename T>
const T sigmoid(const T value){
	return T(1)/(T(1) + expf(-value));
}
template<typename T>
const T tanh(const T value){
	T v2 = expf(2*value);
	return (v2-T(1))/(T(1) + v2);
}

template<int H, int W, typename T>
void matMul(const T w[W][H], const T v[W], T o[H]){
#pragma HLS ARRAY_PARTITION variable=w complete dim=2
#pragma HLS ARRAY_PARTITION variable=o complete dim=1
	for(int i=0;i<H;i++)
#pragma HLS UNROLL
		o[i] = T(0);
	for(int j=0;j<W;j++){
#pragma HLS PIPELINE
		T tmp = v[j];
		for(int i=0;i<H;i++)
			o[i]+=w[j][i] * tmp;
	}
}

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
void lstm(const T input[INPUT_SIZE],
		T state[LSTM_SIZE],
		T output[LSTM_SIZE],
		const T BIAS[LSTM_SIZE * 4],
		const T WI[INPUT_SIZE][LSTM_SIZE * 4],
		const T WS[LSTM_SIZE][LSTM_SIZE * 4]){
	static T hidden[LSTM_SIZE] = {0};
	T gates[LSTM_SIZE * 4];
#pragma HLS ARRAY_PARTITION variable=gates block factor=2
	T out1[LSTM_SIZE * 4];
	T out2[LSTM_SIZE * 4];

	matMul<LSTM_SIZE * 4, INPUT_SIZE>(WI, input, out1);
	matMul<LSTM_SIZE * 4, LSTM_SIZE>(WS, hidden, out2);
//	add<LSTM_SIZE * 4>(out1, out2, BIAS, gates);

#pragma HLS ALLOCATION instances=tanhf limit=1 operation

	for(int i=0;i<LSTM_SIZE;i++){
#pragma HLS PIPELINE
		T g0 = out1[i] + out2[i] + BIAS[i];
		T g1 = out1[LSTM_SIZE + i] + out2[LSTM_SIZE + i] + BIAS[LSTM_SIZE + i];
		T g2 = out1[2 * LSTM_SIZE + i] + out2[2 * LSTM_SIZE + i] + BIAS[2 * LSTM_SIZE + i];
		T g3 = out1[3 * LSTM_SIZE + i] + out2[3 * LSTM_SIZE + i] + BIAS[3 * LSTM_SIZE + i];
		T s = sigmoid(g1) * state[i] + sigmoid(g0)* tanh(g2);
		state[i] =  s;
		hidden[i] = sigmoid(g3) * tanh(s);
		output[i] = hidden[i];
/*
		T s;
		s = sigmoid(gates[LSTM_SIZE + i]) * state[i] + sigmoid(gates[i])* tanh(gates[LSTM_SIZE*2 + i]);
		state[i] =  s;
		output[i] = sigmoid(gates[LSTM_SIZE*3 + i]) * tanh(s);
*/
	}
}


template<int INPUT_SIZE, int OUTPUT_SIZE, typename T>
void feed_forward(const T input[INPUT_SIZE], T output[OUTPUT_SIZE], const T W1[INPUT_SIZE][OUTPUT_SIZE],const T BIAS1[OUTPUT_SIZE]){
	T tmp[OUTPUT_SIZE];
	matMul<OUTPUT_SIZE, INPUT_SIZE>(W1, input, tmp);
	add<OUTPUT_SIZE>(tmp, BIAS1, output);
}

