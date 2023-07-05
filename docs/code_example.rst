Code example
============

Here is a Python function.

.. code-block:: python

    # adavanced Prog
    from datawhispers import advancedProg as ap
    # Example on how to output all mnist numbers from a csv file
    ap.show_mnist_from_file(your_filepath) # your_filepath example data.csv 
    # Example on how to make a regression and output it
    model = ap.Trend(x,y,"polReg", deg=9) 
    model.make_easy_plot("fig1.png")

    #! DataVis
    from datawhispers import datavis as dv 
    analyser = dv.DataVisAnalyse(df, classification_column, high, low)
    analyser.get_all()