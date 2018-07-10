#pragma once
#include <cmath>

template<typename T>
const T sigmoid(const T value){
	return T(1)/(T(1) + std::exp(-value));
}

template<int INPUT_SIZE, int LSTM_SIZE, typename T>
void lstm(const T input[INPUT_SIZE],T state[LSTM_SIZE], T output[LSTM_SIZE], const T BIAS[LSTM_SIZE * 4], const T WI[INPUT_SIZE][LSTM_SIZE * 4], const T WS[LSTM_SIZE][LSTM_SIZE * 4]){

	T gates[LSTM_SIZE * 4];
	for(int i=0;i<LSTM_SIZE * 4; i++){
		T	tmp = BIAS[i];
		for(int j=0;j<INPUT_SIZE;j++){
			tmp+=input[j] * WI[j][i];
		}
		for(int k=0;k<LSTM_SIZE;k++){
			tmp+=output[k] * WS[k][i];
		}
		gates[i] = tmp;
	}
	for(int i=0;i<LSTM_SIZE;i++){
		state[i] = sigmoid(gates[LSTM_SIZE + i]) * state[i] + sigmoid(gates[i])+ tanh(gates[LSTM_SIZE*2 + i]);
		output[i] = sigmoid(gates[LSTM_SIZE*3 + i]) * tanh(state[i]);
	}
}


template<int INPUT_SIZE, int OUTPUT_SIZE, typename T>
void feed_forward(const T input[INPUT_SIZE], T output[OUTPUT_SIZE], const T W1[INPUT_SIZE][OUTPUT_SIZE],const T BIAS1[OUTPUT_SIZE]){
		for(int i=0;i<OUTPUT_SIZE;i++){
			T mid = BIAS1[i];
			for(int j=0;j<INPUT_SIZE;j++){
				mid+=input[j] * W1[j][i];
			}
			output[i] = mid;
	}
}

