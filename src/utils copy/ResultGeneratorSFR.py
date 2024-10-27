import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CENTER = 'CENTER'
F30 = 'F30'
F50 = 'F50'
F60 = 'F60'
F75 = 'F75'
EDGE = 'EDGE'
LEFT_MIDDLE = 'LEFT_MIDDLE'
RIGHT_MIDDLE = 'RIGHT_MIDDLE'
TOP_MIDDLE = 'TOP_MIDDLE'
BOTTOM_MIDDLE = 'BOTTOM_MIDDLE'
LEFT_EDGE = 'LEFT_EDGE'
RIGHT_EDGE = 'RIGHT_EDGE'
TOP_EDGE = 'TOP_EDGE'
BOTTOM_EDGE = 'BOTTOM_EDGE'


class ResultGeneratorSFR:
    def __init__(self, total_output, total_y, figure_save_path):
        self.total_output = total_output
        self.total_y = total_y
        self.figure_save_dir = figure_save_path

    def clear_plt(self, fig_size=(6, 6)):
        plt.cla()
        plt.clf()
        plt.figure(figsize=fig_size)

    def print_acc_by_shot(self):
        self.calc_acc_few_mid_many_sfr()

    def calc_acc_sfr(self):
        diff = abs(self.total_output - self.total_y)
        acc_per = (self.total_y - diff) / self.total_y * 100

        acc_cen, acc_30f, acc_60f, acc_75f, acc_edge = self.split_infd_sfr_output_per_field(acc_per)
        acc_list = [acc_cen, acc_30f, acc_60f, acc_75f, acc_edge]
        field_name = ['CEN', '30F', '60F', '75F', 'EDGE']
        
        result_dict = {}

        for i, field in enumerate(field_name):
            mean_accuracy = np.mean(acc_list[i])
            result_dict[field] = mean_accuracy
            print(f'{field} : {mean_accuracy:.2f}')
            
        return result_dict
            
    def calc_acc_few_mid_many_sfr(self):
        diff = abs(self.total_output - self.total_y)
        acc_per = (self.total_y - diff) / self.total_y * 100

        acc_cen, acc_30f, acc_60f, acc_75f, acc_edge = self.split_infd_sfr_output_per_field(acc_per)
        pred_cen, pred_30f, pred_60f, pred_75f, pred_edge = self.split_infd_sfr_output_per_field(self.total_output)
        real_cen, real_30f, real_60f, real_75f, real_edge = self.split_infd_sfr_output_per_field(self.total_y)
        acc_list = [acc_cen, acc_30f, acc_60f, acc_75f, acc_edge]
        pred_list = [pred_cen, pred_30f, pred_60f, pred_75f, pred_edge]
        real_list = [real_cen, real_30f, real_60f, real_75f, real_edge]
        field_name = ['CEN', '30F', '60F', '75F', 'EDGE']

        for i, field in enumerate(field_name):
            mean_real_y = np.mean(real_list[i], axis=1)
            mean_pred_y = np.mean(pred_list[i], axis=1)
            mean_acc = np.mean(acc_list[i], axis=1)

            min_real_y = np.min(real_list[i], axis=1)
            min_pred_y = np.min(pred_list[i], axis=1)
            min_acc = np.min(acc_list[i], axis=1)

            self.calc_by_metric(field, 'MEAN', mean_real_y, mean_pred_y, mean_acc)
            self.calc_by_metric(field, 'MIN', min_real_y, min_pred_y, min_acc)

    def calc_by_metric(self, field, metric, real_y, pred_y, acc_per):
        low_idx = int(len(real_y) * 0.1)
        high_idx = int(len(real_y) * 0.9)

        real_y_sorted = sorted(real_y)
        low_threshold_val = real_y_sorted[low_idx]
        high_threshold_val = real_y_sorted[high_idx]

        # lower part accuracy
        real_mean_l = np.mean(real_y[real_y <= low_threshold_val])
        pred_mean_l = np.mean(pred_y[real_y <= low_threshold_val])
        acc_l = np.mean(acc_per[real_y <= low_threshold_val])
        print(f"{metric}_[LOW] {field} | real : {real_mean_l:.2f} | pred : {pred_mean_l:.2f} |  acc : {acc_l:.2f}")

        # mid part accuracy
        real_mean_m = np.mean(real_y[(real_y > low_threshold_val) & (real_y <= high_threshold_val)])
        pred_mean_m = np.mean(pred_y[(real_y > low_threshold_val) & (real_y <= high_threshold_val)])
        acc_m = np.mean(acc_per[(real_y > low_threshold_val) & (real_y <= high_threshold_val)])

        print(f"{metric}_[MID] {field} | real : {real_mean_m:.2f} | pred : {pred_mean_m:.2f} | acc : {acc_m:.2f}")

        # higher part accuracy
        real_mean_h = np.mean(real_y[real_y > high_threshold_val])
        pred_mean_h = np.mean(pred_y[real_y > high_threshold_val])
        acc_h = np.mean(acc_per[real_y > high_threshold_val])

        print(f"{metric}_[HIGH] {field} | real : {real_mean_h:.2f} | pred : {pred_mean_h:.2f} | acc : {acc_h:.2f}")

    def plot_histo_all_field_sfr(self, epoch):
        pred_cen, pred_30F, pred_60F, pred_75F, pred_85F = self.split_infd_sfr_output_per_field(self.total_output)
        real_cen, real_30F, real_60F, real_75F, real_85F = self.split_infd_sfr_output_per_field(self.total_y)

        self.plot_histo_per_field_sfr(pred_cen, real_cen, [0.65, 0.8], '00', epoch)
        self.plot_histo_per_field_sfr(pred_30F, real_30F, [0.6, 0.8], '30', epoch)
        self.plot_histo_per_field_sfr(pred_60F, real_60F, [0.5, 0.8], '60', epoch)
        self.plot_histo_per_field_sfr(pred_75F, real_75F, [0.4, 0.8], '75', epoch)
        self.plot_histo_per_field_sfr(pred_85F, real_85F, [0.3, 0.8], '85', epoch)

    def plot_histo_per_field_sfr(self, pred, real, x_lim, field_num, epoch):
        self.clear_plt()
        plt.cla()
        plt.clf()
        plt.hist(pred.mean(axis=1), bins=np.arange(x_lim[0], x_lim[1] + 0.001, 0.001), alpha=0.5,
                 label=f'air SFR {field_num}F')
        plt.hist(real.mean(axis=1), bins=np.arange(x_lim[0], x_lim[1] + 0.001, 0.001), alpha=0.5, label='real val')
        plt.xlim(x_lim)
        plt.ylim([0, 200])
        plt.legend()
        plt.savefig(f"{self.figure_save_dir}/SFR_{field_num}F_epoch{epoch}.png")

    def split_infd_sfr_output_per_field(self, output):
        roi_dict = self.get_infd_sfr_field_kv()
        output_cen = output[:, roi_dict[CENTER]]
        output_f30 = output[:, roi_dict[F30]]
        output_f60 = output[:, roi_dict[F60]]
        output_f75 = output[:, roi_dict[F75]]
        output_edge = output[:, roi_dict[EDGE]]
        return output_cen, output_f30, output_f60, output_f75, output_edge

    def get_infd_sfr_field_kv(self):
        ROIs_center_tan = [137, 138, 139, 140]
        ROIs_30F_tan = [54, 58, 80, 83, 84, 88, 105, 108, 109, 112, 113, 117, 133, 134, 141, 142, 159, 162, 163, 166,
                        167, 171, 188, 191, 192, 196, 218, 222]
        ROIs_60F_tan = [4, 6, 8, 10, 20, 23, 24, 27, 28, 31, 32, 36, 46, 49, 50, 53, 57, 61, 62, 66, 75, 76, 79, 87, 91,
                        92, 100, 101, 104, 116, 120, 121, 126, 129, 130, 145, 146, 150, 154, 155, 158, 170, 174, 175,
                        183, 184, 187, 195, 199, 200, 210, 213, 214, 217, 221, 225, 226, 230, 240, 243, 244, 247, 248,
                        251, 252, 256, 266, 268, 270, 272]
        ROIs_75F_tan = [274, 264, 260, 255, 239, 236, 233, 229, 209, 207, 204, 203, 180, 179, 178, 153, 149, 125, 124,
                        99, 96, 95, 72, 71, 69, 65, 45, 43, 40, 35, 19, 16, 12, 2]
        ROIs_edge_tan = [1, 15, 235, 263, 14, 39, 259, 276]

        ROIs_Left_Middle = [20, 46, 210, 240]
        ROIs_Right_Middle = [36, 66, 230, 256]
        ROIs_Top_Middle = [20, 46, 36, 66]
        ROIs_Bottom_Middle = [210, 240, 230, 256]

        ROIs_Left_Edge = [1, 15, 235, 263]
        ROIs_Right_Edge = [14, 39, 259, 276]
        ROIs_Top_Edge = [1, 15, 14, 39]
        ROIs_Bottom_Edge = [235, 263, 259, 276]

        ROIs_center_tan = np.array(ROIs_center_tan) - 1
        ROIs_30F_tan = np.array(ROIs_30F_tan) - 1
        ROIs_60F_tan = np.array(ROIs_60F_tan) - 1
        ROIs_75F_tan = np.array(ROIs_75F_tan) - 1
        ROIs_edge_tan = np.array(ROIs_edge_tan) - 1

        ROIs_Left_Middle = np.array(ROIs_Left_Middle) - 1
        ROIs_Right_Middle = np.array(ROIs_Right_Middle) - 1
        ROIs_Top_Middle = np.array(ROIs_Top_Middle) - 1
        ROIs_Bottom_Middle = np.array(ROIs_Bottom_Middle) - 1

        ROIs_Left_Edge = np.array(ROIs_Left_Edge) - 1
        ROIs_Right_Edge = np.array(ROIs_Right_Edge) - 1
        ROIs_Top_Edge = np.array(ROIs_Top_Edge) - 1
        ROIs_Bottom_Edge = np.array(ROIs_Bottom_Edge) - 1

        roi_dict = {
            CENTER: ROIs_center_tan,
            F30: ROIs_30F_tan,
            F60: ROIs_60F_tan,
            F75: ROIs_75F_tan,
            EDGE: ROIs_edge_tan,
            LEFT_MIDDLE: ROIs_Left_Middle,
            RIGHT_MIDDLE: ROIs_Right_Middle,
            TOP_MIDDLE: ROIs_Top_Middle,
            BOTTOM_MIDDLE: ROIs_Bottom_Middle,
            LEFT_EDGE: ROIs_Left_Edge,
            RIGHT_EDGE: ROIs_Right_Edge,
            TOP_EDGE: ROIs_Top_Edge,
            BOTTOM_EDGE: ROIs_Bottom_Edge,
        }

        return roi_dict

    def plot_scatter_per_shot_sfr(self, epoch, y_name):
        diff = abs(self.total_output - self.total_y)
        acc_per = (self.total_y - diff) / self.total_y * 100

        sfr_cen, sfr_30F, sfr_60F, sfr_75F, sfr_85F = self.split_infd_sfr_output_per_field(self.total_output)
        acc_per_cen, acc_per_30F, acc_per_60F, acc_per_75F, acc_per_85F = self.split_infd_sfr_output_per_field(acc_per)
        real_sfr_cen, real_sfr_30F, real_sfr_60F, real_sfr_75F, real_sfr_85F = self.split_infd_sfr_output_per_field(
            self.total_y)

        real_field_list = [real_sfr_cen, real_sfr_30F, real_sfr_60F, real_sfr_75F, real_sfr_85F]
        pred_field_list = [sfr_cen, sfr_30F, sfr_60F, sfr_75F, sfr_85F]
        acc_field_list = [acc_per_cen, acc_per_30F, acc_per_60F, acc_per_75F, acc_per_85F]
        fig_name_list = ['Center', '30F', '60F', '75F', '85F']
        fig_size = (20, 3)

        # plt.clf()
        # n_plot = len(fig_name_list) # MAC:4 # INF:5
        # fig, axs = plt.subplots(1, n_plot, figsize=fig_size, sharex=True, sharey=True)

        # for i, real_field in enumerate(real_field_list) :

        #     pred_field = pred_field_list[i]
        #     title = fig_name_list[i]

        #     real_mean = real_field.mean(axis=1)
        #     pred_mean = pred_field.mean(axis=1)

        #     sns.distplot(real_mean, label = 'real', ax=axs[i])
        #     sns.distplot(pred_mean, label = 'pred', ax=axs[i])
        #     axs[i].legend()
        #     axs[i].set_title(title)
        #     axs[i].set_xlim(0.5, 0.8)

        # plt.savefig(self.figure_save_dir + y_name+'_hist_%d.png' % epoch)

        # plt.clf()
        # fig, axs = plt.subplots(1, n_plot, figsize=fig_size, sharex=True, sharey=True)

        # for i, real_field in enumerate(real_field_list) :

        #     pred_field = pred_field_list[i]
        #     title = fig_name_list[i]

        #     real_mean = real_field.mean(axis=1)
        #     pred_mean = pred_field.mean(axis=1)

        #     axs[i].scatter(real_mean, pred_mean)
        #     axs[i].legend()
        #     axs[i].set_title(title)
        #     axs[i].set_xlim(0.5, 0.8)
        #     axs[i].set_ylim(0.5, 0.8)
        #     axs[i].axline((0, 0), slope=1)

        # plt.savefig(self.figure_save_dir + y_name+'_scatter_%d.png' % epoch)

        for i, real_field in enumerate(real_field_list):
            field_name = fig_name_list[i]

            real_y_mean = np.mean(real_field, axis=1)
            pred_y_mean = np.mean(pred_field_list[i], axis=1)
            acc_per_mean = np.mean(acc_field_list[i], axis=1)

            real_y_min = np.min(real_field, axis=1)
            pred_y_min = np.min(pred_field_list[i], axis=1)

            diff_min = abs(real_y_min - pred_y_min)
            acc_per_min = (real_y_min - diff_min) / real_y_min * 100

            self.save_result_by_roi(real_y_mean, pred_y_mean, acc_per_mean, f'mean_{field_name}', epoch)
            self.save_result_by_roi(real_y_min, pred_y_min, acc_per_min, f'min_{field_name}', epoch)

    def save_result_by_roi(self, real_y, pred_y, acc_per, info, epoch):
        low_idx = int(len(real_y) * 0.1)
        high_idx = int(len(real_y) * 0.9)

        real_y_sorted = sorted(real_y)
        low_threshold_val = real_y_sorted[low_idx]
        high_threshold_val = real_y_sorted[high_idx]

        # lower part accuracy
        real_mean_l = np.mean(real_y[real_y <= low_threshold_val])
        pred_mean_l = np.mean(pred_y[real_y <= low_threshold_val])
        # diff_l = np.mean(diff[y_index <= low_idx])
        acc_l = np.mean(acc_per[real_y <= low_threshold_val])

        # low part scatter plot
        self.clear_plt()
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.scatter(real_y[real_y <= low_threshold_val], pred_y[real_y <= low_threshold_val], alpha=0.2)
        plt.xlim([real_y[real_y <= low_threshold_val].min() - 0.05, real_y[real_y <= low_threshold_val].max() + 0.05])
        plt.ylim([real_y[real_y <= low_threshold_val].min() - 0.1, real_y[real_y <= low_threshold_val].max() + 0.1])
        plt.savefig(f"{self.figure_save_dir}/{info}_LOW_{epoch}.png")

        # mid part scatter plot
        self.clear_plt()
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.scatter(real_y[(real_y > low_threshold_val) & (real_y <= high_threshold_val)],
                    pred_y[(real_y > low_threshold_val) & (real_y <= high_threshold_val)], alpha=0.1)
        plt.xlim([real_y[(real_y > low_threshold_val) & (real_y <= high_threshold_val)].min() - 0.05,
                  real_y[(real_y > low_threshold_val) & (real_y <= high_threshold_val)].max() + 0.05])
        plt.ylim([real_y[(real_y > low_threshold_val) & (real_y <= high_threshold_val)].min() - 0.05,
                  real_y[(real_y > low_threshold_val) & (real_y <= high_threshold_val)].max() + 0.05])
        plt.savefig(f"{self.figure_save_dir}/{info}_MED_{epoch}.png")

        # high part scatter plot
        self.clear_plt()
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.scatter(real_y[real_y > high_threshold_val], pred_y[real_y > high_threshold_val], alpha=0.2)
        plt.xlim([real_y[real_y > high_threshold_val].min() - 0.05, real_y[real_y > high_threshold_val].max() + 0.05])
        plt.ylim([real_y[real_y > high_threshold_val].min() - 0.1, real_y[real_y > high_threshold_val].max() + 0.1])
        plt.savefig(f"{self.figure_save_dir}/{info}_HGH_{epoch}.png")
