The Matlab code in this folder solves a canonical RBC model with endogenous labour supply (Frisch) and 
aggregate uncertainty using a global nonlinear solution method in sequence space (Lee, 2024). 

Folder Guide:
./Functions : Folder with auxiliary functions used in the main algorithm.
./Figures : Folder with figures after model is solved.
./RBC_Frisch_GNLS_Slow.m : Solves the model using time loops (slow algorithm). 
./RBC_Frisch_GNLS_Fast.m : Solves the model using vectorization instead of loops (fast algorithm). 
./Notes.pdf : Model details.

Reference:
Lee, H. (2024), "Dynamically Consistent Global Nonlinear Solutions in the Sequence Space: Theory and Applications", Working Paper. 
