It has been 5 days since i heard About deeplearning, and since then I havent been doing anything else than 
coding this MLP. Some important insights and Questions I am Looking for to answer:

Cephei 0 detects circles and distinguishes them from other handrawn objects

Pre-Release 1
-Giving it 50 handrawn Pictures of circles, its only ability lies in differentiating between bright and dark Pictures

-Wrote a script that takes a Picture and saves 100 transformed versions of it (random rotation, enlargement, Translation). This somewhat works, it confidently detects circles, but also mistakes triangles and squares often,
tho never a line. The hypothesis is that Augmentation is not a substitude for diversity, will proceed to draw more Pictures with 10 augmentations for each Picture
 
-Added Validation MSE / Training MSE measurement to DL script

- The hypothesis was true:
Drawing 30 Pictures with 100 augmentations each validation MSE's plateus at 0.1
Drawing 300 Pictures with 10 Augmentation each Validation MSE's plateus at 0.013

Release 
- Same training data
- Added L1 and L2 regularization
- Switched to cross entropy cost function
- Picks the best validation MSE out of all epochs and uses those wheights / biases
- Added possibility for stochastic gradient descent 
New MSE 0.00066 (although validation data was only around 50 pictures)
Due to a lack of training Data this model only works well when the circle is at the center or somewhat there. Im still happy for my first MLP to work well, consider importing MNIST dataset and train it with all the 0's as positives
