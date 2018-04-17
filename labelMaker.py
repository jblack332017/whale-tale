import numpy as np

def getTopFive(a):
  top5 = np.argpartition(a, -5)[-5::]
  return top5[::-1]

def chooseLabels(probabilities, classnames, imagenames, outputPath):
  outFile = open(outputPath, 'w');
  outFile.write("Image,Id\n");
  for i in range(len(probabilities)):
     predictions = getTopFive(probabilities[i]);
     outFile.write(imagenames[i] + ", ");
     names = [];
     for j in range(len(predictions)):
       names.append(classnames[predictions[j]]);
     outFile.write(" ".join(names));
     outFile.write("\n");
  outFile.close();
