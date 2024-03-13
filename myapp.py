import streamlit as st
import json, requests
import sagemaker
from ultralytics import YOLO
import boto3, cv2, time, numpy as np, matplotlib.pyplot as plt, random
from sagemaker.pytorch import PyTorch
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.deserializers import JSONDeserializer
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.pytorch import PyTorchModel
from sagemaker.pytorch import PyTorchPredictor


sm_boto3 = boto3.client("sagemaker",region_name="eu-central-1")
sess = sagemaker.Session()
ENDPOINT = ''

def parse_json_to_yolo(json_data):
    # Implement the logic to parse JSON to YOLO format
    # You may need to convert coordinates, handle classes, etc.
    pass

def predict(input_data):
    # Load a model
    model = YOLO(r"D:\academics\yolov8_unhealthy_trees\runs\segment\train8\weights\best.pt")  # pretrained YOLOv8n model

    # Run batched inference on a list of images
    results = model.predict(r"C:\Users\nadee\Downloads\u3s4k1hylelkfcx9s7_output_196.jpg", save=True, conf=0.5) # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        result.show()  # display to screen
        result.save(filename='result.jpg')  # save to disk

def yolo_inference(yolo_data):
    
    sm_client = boto3.client(service_name="sagemaker",region_name="eu-central-1")

    ENDPOINT_NAME = ENDPOINT
    print(f'Endpoint Name: {ENDPOINT_NAME}')

    endpoint_created = False
    while True:
        response = sm_client.list_endpoints()
        for ep in response['Endpoints']:
            print(f"Endpoint Status = {ep['EndpointStatus']}")
            if ep['EndpointName']==ENDPOINT_NAME and ep['EndpointStatus']=='InService':
                endpoint_created = True
                break
        if endpoint_created:
            break
        time.sleep(5)
        
    predictor = PyTorchPredictor(endpoint_name=ENDPOINT_NAME,
                                deserializer=JSONDeserializer())

    infer_start_time = time.time()

    # orig_image = cv2.imread('dataset/images/train/u3s4k1hylelkfcx9ts_output_183.jpg')
    # orig_image = cv2.imread('u3s4k1hylelkfcx9s7_output_196.jpg')
    response = requests.get("https://www.neuraldesigner.com/images/tree-wilt.webp")
    orig_image = cv2.imdecode(np.frombuffer(response.content, np.uint8), 1)

    image_height, image_width, _ = orig_image.shape
    model_height, model_width = 300, 300
    x_ratio = image_width/model_width
    y_ratio = image_height/model_height

    resized_image = cv2.resize(orig_image, (model_height, model_width))
    payload = cv2.imencode('.jpg', resized_image)[1].tobytes()
    result = predictor.predict(payload)
    # print("result: ", result['boxes'])
    infer_end_time = time.time()

    print(f"Inference Time = {infer_end_time - infer_start_time:0.4f} seconds")

    if 'boxes' in result:
        for idx,(x1,y1,x2,y2,conf,lbl) in enumerate(result['boxes']):
            # Draw Bounding Boxes
            x1, x2 = int(x_ratio*x1), int(x_ratio*x2)
            y1, y2 = int(y_ratio*y1), int(y_ratio*y2)
            color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
            cv2.rectangle(orig_image, (x1,y1), (x2,y2), color, 4)
            cv2.putText(orig_image, f"Class: {int(lbl)}", (x1,y1-40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.putText(orig_image, f"Conf: {int(conf*100)}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            if 'masks' in result:
                # Draw Masks
                mask = cv2.resize(np.asarray(result['masks'][idx]), dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
                for c in range(3):
                    orig_image[:,:,c] = np.where(mask>0.5, orig_image[:,:,c]*(0.5)+0.5*color[c], orig_image[:,:,c])

    if 'probs' in result:
        # Find Class
        lbl = result['probs'].index(max(result['probs']))
        color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
        cv2.putText(orig_image, f"Class: {int(lbl)}", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
    if 'keypoints' in result:
        # Define the colors for the keypoints and lines
        keypoint_color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
        line_color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))

        # Define the keypoints and the lines to draw
        # keypoints = keypoints_array[:, :, :2]  # Ignore the visibility values
        lines = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Torso
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]

        # Draw the keypoints and the lines on the image
        for keypoints_instance in result['keypoints']:
            # Draw the keypoints
            for keypoint in keypoints_instance:
                if keypoint[2] == 0:  # If the keypoint is not visible, skip it
                    continue
                cv2.circle(orig_image, (int(x_ratio*keypoint[:2][0]),int(y_ratio*keypoint[:2][1])), radius=5, color=keypoint_color, thickness=-1)

            # Draw the lines
            for line in lines:
                start_keypoint = keypoints_instance[line[0]]
                end_keypoint = keypoints_instance[line[1]]
                if start_keypoint[2] == 0 or end_keypoint[2] == 0:  # If any of the keypoints is not visible, skip the line
                    continue
                cv2.line(orig_image, (int(x_ratio*start_keypoint[:2][0]),int(y_ratio*start_keypoint[:2][1])),(int(x_ratio*end_keypoint[:2][0]),int(y_ratio*end_keypoint[:2][1])), color=line_color, thickness=2)

    plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
    plt.show()

def deployModel(artifact):

    model_data = f'{artifact}'
    # model_data = 's3://sagemaker-eu-central-1-687462152766/pytorch-training-2023-11-30-23-09-47-893/output/model.tar.gz'
    # model_data = 's3://mlops.webapp/model.tar.gz'

    model = PyTorchModel(entry_point='inference.py',
                        source_dir='./src/',
                        model_data=model_data, 
                        framework_version='1.13.1', 
                        py_version='py39',
                        role="arn:aws:iam::687462152766:role/service-role/AmazonSageMaker-ExecutionRole-20220727T002967",
                        env={'TS_MAX_RESPONSE_SIZE':'20000000'},
                        sagemaker_session=sess,
                        model_server_workers = 2)

    predictor = model.deploy(initial_instance_count=1, 
                            instance_type="ml.m5.2xlarge",
                            deserializer=JSONDeserializer(),
                            )
    global ENDPOINT
    ENDPOINT = predictor.endpoint_name


def train_model():
    sm_boto3 = boto3.client("sagemaker",region_name="eu-central-1")
    sess = sagemaker.Session()
    region = sess.boto_session.region_name
    bucket = 'mlops.webapp' 
    data_location  =  f's3://{bucket}'

    input  =  { 
        'cfg' :  data_location + '/cfg' ,  
        #'weights': data_location+'/weights',  
        'images' :  data_location + '/images' ,  
        'labels' :  data_location + '/labels' } 

    print(input)

    hyperparameters = {'data': '/opt/ml/input/data/cfg/satellite-seg_new.yaml', 
                    'project': '/opt/ml/model/',
                    'name': 'satellite-seg', 
                    'imgsz': 640,
                    'batch': 1,
                    'epochs': 2,
                    'workers':1}

    metric_definitions = [{'Name': 'mAP50',
                        'Regex': '^all\s+(?:[\d.]+\s+){4}([\d.]+)'}]

    yolo_estimator = PyTorch(entry_point='train.py',
                                source_dir='./src/',
                                role="arn:aws:iam::687462152766:role/service-role/AmazonSageMaker-ExecutionRole-20220727T002967",
                                hyperparameters=hyperparameters,
                                framework_version='1.13.1',
                                py_version='py39',
                                script_mode=True,
                                instance_count=1,
                                metric_definitions=metric_definitions,
                                instance_type="ml.m5.2xlarge",
                                use_spot_instances = True,
                                max_wait = 7200,
                                max_run = 3600
                )

    yolo_estimator.fit(input)
    yolo_estimator.latest_training_job.wait(logs="None")
    artifact = sm_boto3.describe_training_job(TrainingJobName=yolo_estimator.latest_training_job.name)["ModelArtifacts"]["S3ModelArtifacts"]
    print("Model artifact persisted at " + artifact)

    deployModel(artifact=artifact)



def main():
    st.set_page_config(page_title="MLops")
    st.title("MLops- DBSE Scientific Project")

    # Create tabs for testing and training
    tab = st.sidebar.selectbox("Select Tab", ["Testing", "Training"])

    if tab == "Testing":
        # Testing tab logic
        st.header("Testing")

        # File upload
        uploaded_file = st.file_uploader("Upload Image for testing",  type=["jpg", "jpeg", "png"])
        predict_button = st.button("Predict")
        print(uploaded_file)
        if uploaded_file is not None and predict_button :
            # json_data = json.load(uploaded_file)

            # # Process JSON data to YOLO format
            # yolo_format_data = parse_json_to_yolo(json_data)

            # Perform YOLO inference
            results = predict(uploaded_file)

            # Display results
            st.subheader("Results")
            st.write(results)

    elif tab == "Training":
        # Training tab logic
        st.header("Training Tab")
        # Add training logic here
        uploaded_file = st.file_uploader("Upload JSON file for training", type=["json"])
    
        train_button = st.button("Train Model")
        if train_button:
            with st.spinner("Training model..."):
                train_model()
                st.success("Model trained successfully!")


if __name__ == "__main__":
    main()
