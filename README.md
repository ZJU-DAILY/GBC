# GBC
Source code of our paper for our paper: Accelerating Biclique Counting on GPU

`BicliquesotaBatch.cu` implements GBC without HTB and work steal. 

`BicliquesotaStealworkBatch.cu` implements GBC without HTB.

`BiclqiueBitmap.cu` implements GBC without work steal.

`BicliqueBitmapStealwork.cu` implements GBC with all techniques. 

`vertex_reorder.cpp` implements Border. 

To compile these files except `vertex_reorder.cpp`, please run the command:
`nvcc BicliquesotaBatch.cu -o bst`

The bst needs four arguments:
`./bst filename p q selected_layer`

For example, `./bst github.txt 8 8 1` denotes calculating the (8,8)-bicliques with the right layer selected. 
