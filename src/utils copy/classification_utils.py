import numpy as np


def get_roi_dict():
    '''
        ROI 영역별 Index를 return 하는 함수.
    '''
    ROIs_center = [137,138,139,140]
    ROIs_30F = [54,58,80,83,84,88,105,108,109,112,113,117,133,134,141,142,159,162,163,166,167,171,188,191,192,196,218,222]
    ROIs_60F = [4,6,8,10,20,23,24,27,28,31,32,36,46,49,50,53,57,61,62,66,75,76,79,87,91,92,100,101,104,116,120,121,126,129,130,145,146,150,154,155,158,170,174,175,183,184,187,195,199,200,210,213,214,217,221,225,226,230,240,243,244,247,248,251,252,256,266,268,270,272]
    ROIs_75F = [274,264,260,255,239,236,233,229,209,207,204,203,180,179,178,153,149,125,124,99,96,95,72,71,69,65,45,43,40,35,19,16,12,2]
    ROIs_edge = [1,15,235,263,14,39,259,276]

    ROIs_Left_Middle = [20,46,210,240]
    ROIs_Right_Middle = [36,66,230,256]
    ROIs_Top_Middle = [20,46,36,66]
    ROIs_Bottom_Middle = [210,240,230,256]

    ROIs_Left_Edge = [1,15,235,263]
    ROIs_Right_Edge = [14,39,259,276]
    ROIs_Top_Edge = [1,15,14,39]
    ROIs_Bottom_Edge = [235,263,259,276]

    ROIs_center = np.array(ROIs_center) - 1
    ROIs_30F = np.array(ROIs_30F) - 1
    ROIs_60F = np.array(ROIs_60F) - 1
    ROIs_75F = np.array(ROIs_75F) - 1
    ROIs_edge = np.array(ROIs_edge) - 1

    ROIs_Left_Middle = np.array(ROIs_Left_Middle) - 1
    ROIs_Right_Middle = np.array(ROIs_Right_Middle) - 1
    ROIs_Top_Middle = np.array(ROIs_Top_Middle) - 1
    ROIs_Bottom_Middle = np.array(ROIs_Bottom_Middle) - 1

    ROIs_Left_Edge = np.array(ROIs_Left_Edge) - 1
    ROIs_Right_Edge = np.array(ROIs_Right_Edge) - 1
    ROIs_Top_Edge = np.array(ROIs_Top_Edge) - 1
    ROIs_Bottom_Edge = np.array(ROIs_Bottom_Edge) - 1

    roi_dict = {
        "CENTER" : ROIs_center,
        "30F" : ROIs_30F,
        "60F" : ROIs_60F,
        "75F" : ROIs_75F,
        "EDGE" : ROIs_edge,
        "LEFT_MIDDLE" : ROIs_Left_Middle,
        "RIGHT_MIDDLE" : ROIs_Right_Middle,
        "TOP_MIDDLE" : ROIs_Top_Middle,
        "BOTTOM_MIDDLE" : ROIs_Bottom_Middle,
        "LEFT_EDGE" : ROIs_Left_Edge,
        "RIGHT_EDGE" : ROIs_Right_Edge,
        "TOP_EDGE" : ROIs_Top_Edge,
        "BOTTOM_EDGE" : ROIs_Bottom_Edge,
    }

    return roi_dict

def classification_label_generator(y_data: np.array, roi_dict: dict):
    '''
        y_data: OUT_000 ~ OUT_275 column 값 (batch_size, y dimension)
        roi_dict: center/30f/60f/75f/edge등 각 영역의 index가 저장된 dictionary
    '''
    REQUIREMENTS_SPEC_DICT = {
        "CENTER_MIN" : 0.65,
        "30F_MIN" : 0.59,
        "60F_MIN" : 0.38,
        "75F_MIN" : 0.25,
        "EDGE_MIN" : 0.15,
        "LR_MIDDLE_DELTA" : [-0.16, 0.16],
        "TB_MIDDLE_DELTA" : [-0.16, 0.16],
        "LR_EDGE_DELTA" : [-0.3, 0.3],
        "TB_EDGE_DELTA" : [-0.3, 0.3],
    }

    label = np.ones(len(y_data), dtype=np.int64)

    value_of_center = y_data[:, roi_dict["CENTER"]]
    value_of_30f = y_data[:, roi_dict["30F"]]
    value_of_60f = y_data[:, roi_dict["60F"]]
    value_of_75f = y_data[:, roi_dict["75F"]]
    value_of_edge = y_data[:, roi_dict["EDGE"]]

    value_of_left_mid = y_data[:, roi_dict["LEFT_MIDDLE"]]
    value_of_right_mid = y_data[:, roi_dict["RIGHT_MIDDLE"]]
    value_of_lr_mid_delta = value_of_left_mid.mean(axis=1) - value_of_right_mid.mean(axis=1)

    value_of_top_mid = y_data[:, roi_dict["TOP_MIDDLE"]]
    value_of_bottom_mid = y_data[:, roi_dict["BOTTOM_MIDDLE"]]
    value_of_tb_mid_delta = value_of_top_mid.mean(axis=1) - value_of_bottom_mid.mean(axis=1)

    value_of_left_edge = y_data[:, roi_dict["LEFT_EDGE"]]
    value_of_right_edge = y_data[:, roi_dict["RIGHT_EDGE"]]
    value_of_lr_edge_delta = value_of_left_edge.mean(axis=1) - value_of_right_edge.mean(axis=1)

    value_of_top_edge = y_data[:, roi_dict["TOP_EDGE"]]
    value_of_bottom_edge = y_data[:, roi_dict["BOTTOM_EDGE"]]
    value_of_tb_edge_delta = value_of_top_edge.mean(axis=1) - value_of_bottom_edge.mean(axis=1)


    label = label & (np.min(value_of_center, axis=1) > REQUIREMENTS_SPEC_DICT["CENTER_MIN"]) # CENTER SPEC
    label = label & (np.min(value_of_30f, axis=1) > REQUIREMENTS_SPEC_DICT["30F_MIN"]) # 30F SPEC
    label = label & (np.min(value_of_60f, axis=1) > REQUIREMENTS_SPEC_DICT["60F_MIN"]) # 60F SPEC
    label = label & (np.min(value_of_75f, axis=1) > REQUIREMENTS_SPEC_DICT["75F_MIN"]) # 75F SPEC
    label = label & (np.min(value_of_edge, axis=1) > REQUIREMENTS_SPEC_DICT["EDGE_MIN"]) # EDGE SPEC

    label = label & ((value_of_lr_mid_delta > REQUIREMENTS_SPEC_DICT["LR_MIDDLE_DELTA"][0]) & \
                     (value_of_lr_mid_delta < REQUIREMENTS_SPEC_DICT["LR_MIDDLE_DELTA"][1])) # MIDDLE DELTA LEFT RIGHT SPEC
    label = label & ((value_of_tb_mid_delta > REQUIREMENTS_SPEC_DICT["TB_MIDDLE_DELTA"][0]) & \
                     (value_of_tb_mid_delta < REQUIREMENTS_SPEC_DICT["TB_MIDDLE_DELTA"][1])) # MIDDLE DELTA TOP BOTTOM SPEC
    label = label & ((value_of_lr_edge_delta > REQUIREMENTS_SPEC_DICT["LR_EDGE_DELTA"][0]) & \
                     (value_of_lr_edge_delta < REQUIREMENTS_SPEC_DICT["LR_EDGE_DELTA"][1])) # EDGE DELTA LEFT RIGHT SPEC
    label = label & ((value_of_tb_edge_delta > REQUIREMENTS_SPEC_DICT["TB_EDGE_DELTA"][0]) & \
                     (value_of_tb_edge_delta < REQUIREMENTS_SPEC_DICT["TB_EDGE_DELTA"][1])) # EDGE DELTA TOP BOTTOM SPEC
    
    return label


if __name__== "__main__":
    batch_size = 256
    y_dim = 276

    dummy_Y_data = np.random.rand(batch_size, y_dim)

    roi_dict = get_roi_dict()

    label = classification_label_generator(dummy_Y_data, roi_dict)