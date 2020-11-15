# This is a copy of https://github.com/streamlit/demo-self-driving/blob/master/streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
import urllib
import altair as alt

DATA_URL_ROOT = "https://streamlit-self-driving.s3-us-west-2.amazonaws.com/"

# External files to download.
EXTERNAL_DEPENDENCIES = {
    "yolov3.weights": {
        "url": "https://pjreddie.com/media/files/yolov3.weights",
        "size": 248007048
    },
    "yolov3.cfg": {
        "url": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        "size": 8342
    }
}

def main():
    run_app()


def get_selected_frames(summary, label, min_elements, max_elements):
    return summary[np.logical_and(summary[label] >= min_elements, summary[label] < max_elements)].index


def frame_selector_ui(summary):
    st.sidebar.markdown('# Frame')

    object_type = st.sidebar.selectbox('Search for which objects?', summary.columns, 2)

    min_elements, max_elements = st.sidebar.slider(f"How many {object_type} ? Select a range", 0, 25, [10, 20])

    selected_frames = get_selected_frames(summary, object_type, min_elements, max_elements)
    if len(selected_frames) < 1:
        return None, None

    selected_frame_index = st.sidebar.slider("Choose a frame (index)", 0, len(selected_frames) - 1, 0)
    objects_per_frame = summary.loc[selected_frames, object_type].reset_index(drop=True).reset_index()

    #Below part is confusing
    chart = alt.Chart(objects_per_frame, height=20).mark_area().encode(
        alt.X('index:Q', scale=alt.Scale(nice=False)),
        alt.Y(f'{object_type}:Q'))

    selected_frame_df = pd.DataFrame({'selected_frame': [selected_frame_index]})

    vline = alt.Chart(selected_frame_df).mark_rule(color='red').encode(x = "selected_frame")
    st.sidebar.altair_chart(alt.layer(chart, vline))

    selected_frame = selected_frames[selected_frame_index]
    return selected_frame_index, selected_frame


def object_detector_ui():
    st.sidebar.markdown('# Model')
    confidence_threshold = st.sidebar.slider('Confidence threshold', 0.0, 1.0, 0.5, 0.01)
    overlap_threshold = st.sidebar.slider('Overlap threshold', 0.0, 1.0, 0.3, 0.01)
    return confidence_threshold, overlap_threshold


def load_image(image_url):
    with urllib.request.urlopen(image_url) as response:
        image = np.asarray(bytearray(response.read()), dtype='uint8')
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = image[:, :, [2,1, 0]] # BGR -> RGB
        return image


def draw_image_with_boxes(image, boxes, header, description):
    LABEL_COLORS = {
        'car': [255, 0, 0],
        'pedestrian': [0, 255, 0],
        'truck': [0, 0, 255],
        'trafficLight': [255, 255, 0],
        'biker': [255, 0, 255]
    }
    image_with_boxes = image.astype(np.float64)

    for _, (xmin, ymin, xmax, ymax, label) in boxes.iterrows():
        image_with_boxes[int(ymin):int(ymax), int(xmin):int(xmax), :] += LABEL_COLORS[label]
        image_with_boxes[int(ymin):int(ymax), int(xmin):int(xmax), :] /= 2

    st.subheader(header)
    st.markdown(description)
    st.image(image_with_boxes.astype(np.uint8), use_column_width=True)


def yolo_v3(image, confidence_threshold, overlap_threshold):
    @st.cache(allow_output_mutation=True)
    def load_network(config_path, weights_path):
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        output_layer_names = net.getLayerNames()
        output_layer_names = [output_layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return net, output_layer_names

    net, output_layer_names = load_network('yolov3.cfg', 'yolov3.weights')

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layer_names)

    bboxes, confidences, class_ids = [], [], []

    H, W = image.shape[:2]
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                box = detection[:4] * np.array([W, H, H, W])
                centerX, centerY, width, height = box.astype("int")
                x, y = int(centerX - (width/2)), int(centerY - (height/2))
                bboxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(bboxes, confidences, confidence_threshold, overlap_threshold)

    UDACITY_LABELS = {
        0: 'pedestrian',
        1: 'biker',
        2: 'car',
        3: 'biker',
        5: 'truck',
        7: 'truck',
        9: 'trafficLight'
    }
    xmin, xmax, ymin, ymax, labels = [], [], [], [], []
    if len(indices) > 0:
        # loop over the indexes we are keeping
        for i in indices.flatten():
            label = UDACITY_LABELS.get(class_ids[i], None)
            if label is None:
                continue

            # extract the bounding box coordinates
            x, y, w, h = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]

            xmin.append(x)
            ymin.append(y)
            xmax.append(x+w)
            ymax.append(y+h)
            labels.append(label)

    boxes = pd.DataFrame({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "labels": labels})
    return boxes[["xmin", "ymin", "xmax", "ymax", "labels"]]



def run_app():

    @st.cache
    def load_metadata(url):
        return pd.read_csv(url)

    @st.cache
    def create_summary(metadata):
        one_hot_encoded = pd.get_dummies(metadata[['frame', 'label']], columns=['label'])

        summary = one_hot_encoded.groupby(['frame']).sum().rename(columns={
            'label_biker': 'biker',
            'label_car': 'car',
            'label_trafficLight': 'traffic light',
            'label_truck': 'truck',
            'label_pedestrian': 'pedestrian'
        })

        return summary

    st.write("""
    This sample shows how to build a simple image and model prediction browser
    - Use cv2 load yolov3
    - render image on the UI
    - The use of siderbar and slider is good
    
    It's more like a web application.
    """)

    metadata = load_metadata(os.path.join(DATA_URL_ROOT, 'labels.csv.gz'))
    summary = create_summary(metadata)
    selected_frame_index, selected_frame = frame_selector_ui(summary)

    if selected_frame_index == None:
        st.error('No frames fit the criteria')
        return

    confidence_threshold, overlap_threshold = object_detector_ui()

    image_url = os.path.join(DATA_URL_ROOT, selected_frame)
    image = load_image(image_url)

    bboxes = metadata[metadata.frame == selected_frame].drop(columns=['frame'])
    draw_image_with_boxes(image, bboxes, "Ground Truth", f"Human Annotated Image (frame: {selected_frame_index})")

    yolo_bboxes = yolo_v3(image, confidence_threshold, overlap_threshold)

    draw_image_with_boxes(image, yolo_bboxes, "Yolo Detection", f"Yolo detection image (frame: {selected_frame_index})")


if __name__ == '__main__':
    main()