#include "lstm_hls.h"
#include "params.h"
void position(const DataType input[DataSize][Capa_In_Size + Infrared_In_Size],
		DataType p[DataSize][Output_Size]){
#pragma HLS INTERFACE m_axi depth=393*20 port=input bundle=gmem
#pragma HLS INTERFACE m_axi depth=393*2 port=input bundle=gmem
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=p bundle=control

	static const int InputSize = Capa_In_Size + Infrared_In_Size;
	static const int LSTM_SIZE = 16;
	
	static const DataType WI[InputSize][LSTM_SIZE * 4] ={
#include "expr_0.txt"
	};
	static const DataType WS[LSTM_SIZE][LSTM_SIZE * 4] = {
#include "expr_1.txt"
	};
	static const DataType BiasS[LSTM_SIZE * 4]={
#include "expr_2.txt"
	};

	static const DataType W1[LSTM_SIZE][Output_Size] ={
#include "expr_3.txt"
	};
	static const DataType Bias1[Output_Size]={
#include "expr_4.txt"
	};

	DataType state[LSTM_SIZE] = {0
	};

	for(int i=0; i< DataSize;i++){
#pragma HLS DATAFLOW
		DataType output[LSTM_SIZE];
		lstm<InputSize, LSTM_SIZE>(input[i], state, output, BiasS, WI, WS);
		feed_forward<LSTM_SIZE, Output_Size>(output, p[i], W1, Bias1);
	}
}
