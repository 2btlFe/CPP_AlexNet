%Get training images
flower_ds = imageDatastore('Flowers','IncludeSubfolders',true,'LabelSource','foldernames');
[trainImgs,testImgs] = splitEachLabel(flower_ds,0.6);% splitEachLabel(flower_ds,0.6, ‘randomize’);
numClasses = numel(categories(flower_ds.Labels));

%Create a network by modifying AlexNet
net = alexnet;
layers = net.Layers;
layers(end-2) = fullyConnectedLayer(numClasses);
layers(end) = classificationLayer;

%Set training algorithm options
options = trainingOptions('sgdm','InitialLearnRate', 0.001,'MaxEpochs', 20, 'MiniBatchSize',64);

%Perform training
[flowernet,info] = trainNetwork(trainImgs, layers, options);

%Use trained network to classify test images
testpreds = classify(flowernet,testImgs);

Accuracy=mean(testpreds== testImgs.Labels)

