import altair as alt
#alt.renderers.enable('vegascope')
#alt.renderers.enable('notebook')
# load a simple dataset as a pandas DataFrame
from vega_datasets import data
cars = data.cars()

chart = alt.Chart(cars).mark_point().encode(
    x='Horsepower',
    y='Miles_per_Gallon',
    color='Origin',
).interactive()

chart.show()