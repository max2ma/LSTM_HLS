#include "lstm_hls.h"
#include "params.h"
void position(const float input[DataSize][Capa_In_Size + Infrared_In_Size], float p[DataSize][Output_Size]){
	static const int InputSize = Capa_In_Size + Infrared_In_Size;
	static const int LSTM_SIZE = 16;
	
	static const float WI[InputSize][LSTM_SIZE * 4] ={
#include "expr_0.txt"
	};
	static const float WS[LSTM_SIZE][LSTM_SIZE * 4] = {
#include "expr_1.txt"
	};
	static const float BiasS[LSTM_SIZE * 4]={
#include "expr_2.txt"
	};

	static const float W1[LSTM_SIZE][Output_Size] ={
#include "expr_3.txt"
	};
	static const float Bias1[Output_Size]={
#include "expr_4.txt"
	};

	float state[LSTM_SIZE] = {0
	};
	float output[LSTM_SIZE] = {0
	};
	for(int i=0; i< DataSize;i++){
		lstm<InputSize, LSTM_SIZE>(input[i], state, output, BiasS, WI, WS);
		feed_forward<LSTM_SIZE, Output_Size>(output, p[i], W1, Bias1);
	}
}

