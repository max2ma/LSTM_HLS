#include "lstm_hls.h"
#include "params.h"
#include "hls_stream.h"

void position(const int Samples, const DataType (*input)[InputSize], DataType  (*p)[Output_Size]){
#pragma HLS INTERFACE m_axi depth=500 port=input bundle=gmem
#pragma HLS INTERFACE m_axi depth=50 port=p bundle=gmem
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=p bundle=control
#pragma HLS INTERFACE s_axilite port=samples bundle=control

	//static const int InputSize = Capa_In_Size + Infrared_In_Size;
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

#pragma HLS DATAFLOW
	hls::stream<DataType> out_stream;
#pragma HLS STREAM variable=out_stream depth=LSTM_SIZE dim=1
	lstm<InputSize, LSTM_SIZE>(input, out_stream, BiasS, WI, WS,Samples);
	feed_forward<LSTM_SIZE, Output_Size>(out_stream, p, W1, Bias1,Samples);
}
