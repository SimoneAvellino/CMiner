# Brute force testing

Executing the `main.py` will run the following tests:

1. Generate a random connected graph G with n nodes
2. Create a database DB with many graphs, each one is G
3. Run the script `CMiner DB 0.5 -l n -u n`
4. Test if the solution is only G
5. Repeat steps 1-4 for k times

If the mining algorithm is correct, the solution should be only G. 
If the solution is different from G, the algorithm is incorrect.