import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import time

st.title('Example in Guanzhong')
st.write("""
Basically, in this tutorial, streamlit shows its ability to:
- Display Message and raw pandas dataframe table
- Display chart using naive streamlit function, matplotlib, plotly and pydeck
- Interactive streamlit
- Change the default layout (Add sidebar etc.)
- Show some simple and basic components, like button, progress bar, dropdown menu, expander

Through the tutorial, I find streamlit is better than jupyter notebook in:
- A clear split of code and display. I love the streamlit way more.
    - This could help when present the findings. Ignore the code and read directly the results
    - This helps to write a cleaner code
- It's interactive, this means we can present a "pivot table" using interactive and beautiful way
- The default figure by streamlit supports more kinds of interactive

While the concern is:
- The potential performance issue. 
    - The whole page is reloaded when I change the code, which means all code is executed from start. this could lead a performance issue.
- The sharing part. It's good to present, but if I want to deploy to a server and share with certain people, how to handle the auth?
""")

st.write('---')
st.write('## Here the tutorial begin')
st.write('Write a formula using latex')
st.latex(r' a = \frac{1}{n} ')
st.write("Here's our first attempt at using data to create a table:")

first_df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
})

st.write(first_df)

second_df = pd.DataFrame({'second df first column': [1,2,3,4],
                          'second df second column': [10, 20, 30, 40]})

second_df

st.write("### Display Streamlit Chart Below")

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

st.write("### Display Matplotlib Fig with multiple axes below")
st.write('The chart drawn by Matplotlib is displayed as a figure')
fig, axes = plt.subplots(1, 2)

second_df.plot.scatter(x='second df first column', y='second df second column', ax=axes[0])
chart_data.plot(ax=axes[1])

st.pyplot(fig)

st.write('### Display Plotly 3D Scatter Plot Below')
st.write('The 3D Scatter plot is still interactive. Love it!')
trace = go.Scatter3d(x=chart_data['a'], y=chart_data['b'], z = chart_data['c'],
                     mode='markers', marker = {'size':12})
layout = go.Layout(title = '3D Scatter plot by Plotly')
fig = go.Figure(data = [trace], layout = layout)

st.plotly_chart(fig)

st.write('### Display Map')
st.write("""
Display a map with points on it.
This is a wrapper around st.pydeck_chart to quickly create scatterplot charts on top of a map, with auto-centering and auto-zoom.
""")
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)


st.write('### Interactive')
if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    st.line_chart(chart_data)


st.write('If you select a new option, the whole page will be reloaded, the performance is perhaps a concern.')
option = st.sidebar.selectbox(
    'Which number do you like best?',
     first_df['first column'])

'You selected: ', option

left_column, right_column = st.beta_columns(2)
pressed = left_column.button('Press me?')
if pressed:
    right_column.write("You press left column Woohoo!")

expander = st.beta_expander("FAQ")
expander.write("Here you could put in some really, really long explanations...")

'Starting a long computation...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(50):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we\'re done!'

