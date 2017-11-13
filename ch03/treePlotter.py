import matplotlib.pyplot as plt

decisionNode = dict(boxstyle = "sawtooth", fc = "0.8")
leafNode = dict(boxstyle = "round4", fc = "0.8")
arrow_args = dict(arrowstyle = "<-")

def plotNode(nodeTxt, centerPt, parrentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy = parrentPt, xycoords = 'ax')
