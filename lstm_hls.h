#pragma once
#include <cmath>

template<typename T>
const T sigmoid(const T value){
	return T(1)/(T(1) + std::exp(-value));
}

template<int H, int W, typename T>
void matMul(const T w[W][H], const T v[W], T o[H]){
	for(int i=0;i<H;i++){
		T tmp = T(0);
		for(int j=0;j<W;j++)
			tmp+=w[j][i] * v[j];
		o[i] = tmp;
	}
}

template<int N, typename T>
void add(const T a0[N], const T a1[N], T a2[N]){
	for(int i=0;i<N;i++)
		a2[i] = a0[i] + a1[i] ;
}

template<int N, typename T>
void add(const T a0[N], const T a1[N], const T a2[N], T a3[N]){
	for(int i=0;i<N;i++)
		a3[i] = a0[i] + a1[i] + a2[i];
}

template<int INPUT_SIZE, int LSTM_SIZE, typename T>
void lstm(const T input[INPUT_SIZE],T state[LSTM_SIZE], T output[LSTM_SIZE], const T BIAS[LSTM_SIZE * 4], const T WI[INPUT_SIZE][LSTM_SIZE * 4], const T WS[LSTM_SIZE][LSTM_SIZE * 4]){

	T gates[LSTM_SIZE * 4];
	T out1[LSTM_SIZE * 4];
	T out2[LSTM_SIZE * 4];
	
	matMul<LSTM_SIZE * 4, INPUT_SIZE>(WI, input, out1);
	matMul<LSTM_SIZE * 4, LSTM_SIZE>(WS, output, out2);
	add<LSTM_SIZE * 4>(out1, out2, BIAS, gates);

	for(int i=0;i<LSTM_SIZE;i++){
		state[i] = sigmoid(gates[LSTM_SIZE + i]) * state[i] + sigmoid(gates[i])+ tanh(gates[LSTM_SIZE*2 + i]);
		output[i] = sigmoid(gates[LSTM_SIZE*3 + i]) * tanh(state[i]);
	}
}


template<int INPUT_SIZE, int OUTPUT_SIZE, typename T>
void feed_forward(const T input[INPUT_SIZE], T output[OUTPUT_SIZE], const T W1[INPUT_SIZE][OUTPUT_SIZE],const T BIAS1[OUTPUT_SIZE]){
	T tmp[OUTPUT_SIZE];
	matMul<OUTPUT_SIZE, INPUT_SIZE>(W1, input, tmp);
	add<OUTPUT_SIZE>(tmp, BIAS1, output);
}

