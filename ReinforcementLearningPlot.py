import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.widgets as wdg
import multiprocessing as mp
from NeuralNetworks import NeuralNetwork
from math import ceil

class RadioButtons(wdg.RadioButtons):
    def __init__(self, ax, labels, active, active_color='black', size=49, columns=1, loc='center right', **kwargs):
        wdg.AxesWidget.__init__(self, ax)
        self.activecolor = active_color
        self.value_selected = None

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_navigate(False)
        ax.axis('off')

        circles, legend_labels, i = [], [], 0
        labels = labels if isinstance(labels[0], (list, tuple)) else [labels]
        axcolor = ax.get_facecolor()
        for j, label_group in enumerate(labels):
            for label in label_group:
                if i == active: self.value_selected = label
                legend_labels.append(label)
                circles.append(ax.scatter([],[], s=size, marker="o", edgecolor=['black','purple','blue'][j%3], facecolor=active_color if i == active else axcolor))
                i += 1

        if columns > 1:
            aux_labels, aux_circles = [], []
            elements = len(legend_labels)
            rows = int(ceil(elements/columns))
            for j in range(columns):
                for i in range(rows):
                    idx = i*columns+j
                    if idx < elements:
                        aux_labels.append(legend_labels[idx])
                        aux_circles.append(circles[idx])
            legend_labels, circles = aux_labels,aux_circles

        kwargs.setdefault("frameon", False)
        if columns == 0: columns = len(legend_labels)

        self.box = ax.legend(circles, legend_labels, loc=loc, handletextpad=0.0, columnspacing=1.0, scatteryoffsets=[0.55], ncol=columns, **kwargs)
        self.labels = self.box.texts
        self.circles = self.box.legendHandles
        for c in self.circles: c.set_picker(5)
        self.cnt = 0
        self.observers = {}

        self.connect_event('pick_event', self._clicked)


    def _clicked(self, event):
        if self.ignore(event) or event.mouseevent.button != 1 or event.mouseevent.inaxes != self.ax:
            return
        if event.artist in self.circles:
            self.set_active(self.circles.index(event.artist))

class Plotter:
    def __init__(self, title="", path="", figsize=(16, 9), resolution=30, threaded=True):
        if threaded:
            self.__plotter = Plotter(title, path, figsize, resolution, threaded=False)
            self.__pipe, self.__plotter.__pipe = mp.Pipe()
            self.__threaded, self.__thread_process = True, False
            self.__plotter.__threaded, self.__plotter.__thread_process = True, True
            self.__plot_process = mp.Process(target=self.__plotter.initialize, daemon=True)
            self.__plot_process.start()
        else:
            self.__threaded, self.__thread_process = False, False
            self.title = title
            self.path = path
            self.figsize = figsize
            self.resolution = resolution
            self.__callback = None

    def initialize(self):
        if self.__thread_process:
            self.__initialize()
        elif not self.__threaded:
            self.__initialize()


    def update_trajectory(self, episode, steps, destination, trajectory, act_noise):
        self.__pipe.send(['T', episode, steps, destination, trajectory, act_noise])

    def update_all(self, episode):
        self.__pipe.send(['A', episode])

    def receive_position(self, block=True):
        values = None
        if block:
            while values is None:
                while self.__pipe.poll():
                    values = self.__pipe.recv()
        else:
            while self.__pipe.poll():
                values = self.__pipe.recv()
        return np.array(values if values else (0,0))

    def terminate(self):
        self.__pipe.send(None)

    def __initialize(self):
        self.__P = NeuralNetwork.load('{0:s}episode_{1:07d}.pnet'.format(self.path, 0))
        self.__obs_size = np.prod(self.__P.input_layers[0].out_shape)
        self.__act_size = np.prod(self.__P.output_layers[0].out_shape)
        self.__obs = np.zeros(self.__obs_size)
        self.__act = np.zeros(self.__act_size)
        self.__x_axis, self.__y_axis, self.__z_axis = 0, 1, 0
        self.__percentile_idx, self.__max_min = 0, True
        self.__x_min, self.__x_max = 0, 0
        self.__steps, self.__episode, self.__loaded_episode = 0, 0, 0
        self.__act_noise = 0.0

        self.__colormap = [
            mpl.colors.ListedColormap(mpl.cm.bone(np.linspace(0, 1, 1000))[150:850,:-1]),
            mpl.colors.ListedColormap(mpl.cm.winter(np.linspace(0, 1, 1000))[0:1000, :-1]*0.9),
            mpl.colors.ListedColormap(mpl.cm.gist_heat(np.linspace(0, 1, 1000))[0:900, :-1]),
            mpl.colors.ListedColormap(mpl.cm.summer(np.linspace(0, 1, 1000))[0:800, :-1])
            ][1]

        self.__axis_1D = np.linspace(-1, 1, num=self.resolution)
        self.__mx, self.__my = np.meshgrid(self.__axis_1D, self.__axis_1D)
        mx = np.array(self.__mx).reshape(-1,1)
        my = np.array(self.__my).reshape(-1,1)
        self.__axis_2D = np.concatenate((mx,my), axis=1)

        self.__fig = plt.figure(figsize=self.figsize)
        self.__fig.canvas.set_window_title(self.title)
        self.__axs_text_fixed = self.__fig.add_subplot(100, 100, (1,10000))
        self.__axs_text_fixed.set_xticks([])
        self.__axs_text_fixed.set_yticks([])
        self.__axs_text_fixed.set_navigate(False)
        self.__axs_text_fixed.axis('off')

        self.__axs_text = self.__fig.add_subplot(100, 100, (307,5540))
        self.__axs_T = self.__fig.add_subplot(100, 100, (1408,4337))
        self.__axs_Q = self.__fig.add_subplot(100, 100, ( 338,3267), projection='3d')
        self.__axs_A = self.__fig.add_subplot(100, 100, ( 358,3297), projection='3d')
        axs_R1 = self.__fig.add_subplot(100, 100, (7906,9925))
        axs_R2 = self.__fig.add_subplot(100, 100, (7931,9950), sharex=axs_R1)
        axs_R3 = self.__fig.add_subplot(100, 100, (7956,9975), sharex=axs_R1)
        axs_R4 = self.__fig.add_subplot(100, 100, (7981,10000), sharex=axs_R1)
        self.__axs_R = [axs_R1, axs_R2, axs_R3, axs_R4]
        self.__fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.2, wspace=0.2)
        self.__fig.canvas.mpl_connect("button_press_event", self._trajectory_click_CB)

        self.__txt_obs = wdg.TextBox(plt.axes([0.5, 0.55, 0.25, 0.04]), "Observed State", initial=str(self.__obs))
        self.__txt_obs.on_submit(self._set_obs_CB)
        self.__txt_act = wdg.TextBox(plt.axes([0.8, 0.55, 0.15, 0.04]), "Action", initial=str(self.__act))
        self.__txt_act.on_submit(self._set_act_CB)

        self.__obs_list = ["S{0:d}".format(i) for i in range(self.__obs_size)]
        self.__act_list = ["A{0:d}".format(i) for i in range(self.__act_size)]
        button_list = [self.__obs_list, self.__act_list]
        rows = int(np.ceil((self.__obs_size + self.__act_size)/10))
        self.__axs_text_fixed.text(0.5, 0.53-(0.025*0.5+0.001)*rows, "Variable 1", va='center', ha='right')
        self.__rdb_axis_x = RadioButtons(plt.axes([0.5, 0.53-0.025*rows, 0.45, 0.025*rows]), button_list, 0, columns=10, loc='center left')
        self.__rdb_axis_x.on_clicked(self._set_x_axis_CB)
        self.__axs_text_fixed.text(0.5, 0.52-(0.025*1.5+0.004)*rows, "Variable 2", va='center', ha='right')
        self.__rdb_axis_y = RadioButtons(plt.axes([0.5, 0.52-0.05*rows, 0.45, 0.025*rows]), button_list, 1, columns=10, loc='center left')
        self.__rdb_axis_y.on_clicked(self._set_y_axis_CB)
        button_list = [['Target'], self.__act_list]
        self.__rdb_axis_z = RadioButtons(plt.axes([0.85, 0.60, 0.04, 0.025*(self.__act_size+1)]), button_list, 1, loc='upper left')
        self.__rdb_axis_z.on_clicked(self._set_z_axis_CB)

        button_list = ('3D', 'Top', 'Axis1(F)', 'Axis1(B)', 'Axis2(F)', 'Axis2(B)')
        self.__axs_text_fixed.text(0.5, 0.4975-(0.05+0.007)*rows, "View", va='center', ha='right')
        self.__rdb_view = RadioButtons(plt.axes([0.5, 0.485-0.05*rows, 0.45, 0.025]), button_list, 0, columns=10, loc='center left')
        self.__rdb_view.on_clicked(self._set_view_CB)

        self.__x_span1 = wdg.SpanSelector(axs_R1, self._set_x_span_CB, 'horizontal', minspan=10, useblit=True)
        self.__x_span2 = wdg.SpanSelector(axs_R2, self._set_x_span_CB, 'horizontal', minspan=10, useblit=True)
        self.__x_span3 = wdg.SpanSelector(axs_R3, self._set_x_span_CB, 'horizontal', minspan=10, useblit=True)
        self.__x_span4 = wdg.SpanSelector(axs_R4, self._set_x_span_CB, 'horizontal', minspan=10, useblit=True)

        self.__btn_reset = wdg.Button(plt.axes([0.1, 0.32, 0.07, 0.04]), "Reset Span")
        self.__btn_reset.on_clicked(self._reset)
        self.__btn_max_min = wdg.Button(plt.axes([0.2, 0.32, 0.09, 0.04]), 'Max/Min: ON')
        self.__btn_max_min.on_clicked(self._change_max_min)
        self.__btn_percentile = wdg.Button(plt.axes([0.32, 0.32, 0.09, 0.04]), 'Percentile:   1-99')
        self.__btn_percentile.on_clicked(self._change_percentile)

        self.__update_plot = True
        timer = self.__fig.canvas.new_timer(interval=200)
        timer.add_callback(self._timer_CB)
        timer.start()
        plt.show()

        #self.__QvarsCB = buttonCallback(0, 1, min_value=0, max_value=self.__obs_size+self.__act_size-1, onUpdate=self.__plotQ)
        #self.__btn_Q_up1 = wdg.Button(plt.axes([0.91, 0.8345, 0.02, 0.02]), "◄▲▼►")
        #self.__btn_Q_up1.on_clicked(self.__QvarsCB.value1Up)

        #self.__txt_p_min = wdg.TextBox(plt.axes([0.45, 0.05, 0.025, 0.03]), "Percentile Range ", initial=str(self.__percentile_low))
        #self.__txt_p_min.on_submit(self._set_percentile_low_CB)
        #self.__txt_p_max = wdg.TextBox(plt.axes([0.485, 0.05, 0.025, 0.03]), "- ", initial=str(self.__percentile_high))
        #self.__txt_p_max.on_submit(self._set_percentile_high_CB)

    def _trajectory_click_CB(self, event):
        if event.inaxes == self.__axs_T:
            rel_pos = self.__axs_T.transAxes.inverted().transform((event.x, event.y))
            xlim = self.__axs_T.get_xlim()
            ylim = self.__axs_T.get_ylim()
            self.__pipe.send((rel_pos[0]*(xlim[1]-xlim[0])+xlim[0], rel_pos[1]*(ylim[1]-ylim[0])+ylim[0]))

    def _set_view_CB(self, label):
        if '3D' == label:
            self.__axs_Q.view_init()
            self.__axs_A.view_init()
        elif 'Top' == label:
            self.__axs_Q.view_init(90, 0)
            self.__axs_A.view_init(90, 0)
        elif 'Axis1(F)' == label:
            self.__axs_Q.view_init(0, 90)
            self.__axs_A.view_init(0, 90)
        elif 'Axis1(B)' == label:
            self.__axs_Q.view_init(0, -90)
            self.__axs_A.view_init(0, -90)
        elif 'Axis2(F)' == label:
            self.__axs_Q.view_init(0, 0)
            self.__axs_A.view_init(0, 0)
        elif 'Axis2(B)' == label:
            self.__axs_Q.view_init(0, -180)
            self.__axs_A.view_init(0, -180)
        self.__plotQ()
        self.__plotA()

    def _set_x_axis_CB(self, label):
        for i, name in enumerate(self.__obs_list + self.__act_list):
            if name == label:
                self.__x_axis = i
        self.__plotQ()
        self.__plotA()

    def _set_y_axis_CB(self, label):
        for i, name in enumerate(self.__obs_list + self.__act_list):
            if name == label:
                self.__y_axis = i
        self.__plotQ()
        self.__plotA()

    def _set_z_axis_CB(self, label):
        if label == 'Target':
            self.__z_axis = -1
        for i, name in enumerate(self.__act_list):
            if name == label:
                self.__z_axis = i
        self.__plotA()

    def __get_axes_variables(self):
        y_axis = self.__y_axis
        if self.__x_axis == y_axis:
            y_axis = y_axis+1 if y_axis+1 < self.__obs_size+self.__act_size else 0
        return self.__x_axis, y_axis, self.__z_axis

    def _set_obs_CB(self, text):
        obs = self.__obs
        try:
            obs = np.fromstring(text[1:-1], dtype=np.float, sep=' ')
        except (ValueError, TypeError):
            pass
        if obs.shape[0] == self.__obs_size: self.__obs = obs
        self.__txt_obs.set_val(str(self.__obs))
        self.__plotQ()
        self.__plotA()

    def _set_act_CB(self, text):
        act = self.__act
        try:
            act = np.fromstring(text[1:-1], dtype=np.float, sep=' ')
        except (ValueError, TypeError):
            pass
        if act.shape[0] == self.__act_size: self.__act = act
        self.__txt_act.set_val(str(self.__act))
        self.__plotQ()
        self.__plotA()

    def _set_x_span_CB(self, x_min, x_max):
        self.__x_min = int(x_min)
        self.__x_max = int(np.ceil((x_max-x_min)/200) * 200 + self.__x_min) if x_max-x_min > 200 else int(x_max)
        self.__plotRewards()

    def _reset(self, event):
        self.__x_min = 0
        self.__x_max = self.__loaded_episode
        self.__plotRewards()

    def _change_max_min(self, event):
        self.__max_min = False if self.__max_min else True
        self.__btn_max_min.label.set_text('Max/Min: ' + ('ON' if self.__max_min else 'OFF'))
        self.__plotRewards()

    def _change_percentile(self, event):
        button_list = ('  1-99', '  5-95', '10-90', '25-75')
        self.__percentile_idx = self.__percentile_idx + 1 if self.__percentile_idx < 3 else 0
        self.__btn_percentile.label.set_text('Percentile: ' + button_list[self.__percentile_idx])
        self.__plotRewards()

    def _timer_CB(self):
        while self.__pipe.poll():
            values = self.__pipe.recv()
            if values is None:
                plt.close('all')
                return False
            else:
                if values[0] == 'T':
                    self.__episode = values[1]
                    self.__steps = values[2]
                    self.__plotTrajectory(values[3], values[4])
                    self.__act_noise = values[5]
                    self.__plotParameters()

                elif values[0] == 'A':
                    self.__episode = values[1]

                    self.__P = NeuralNetwork.load('{0:s}episode_{1:07d}.pnet'.format(self.path, self.__episode))
                    self.__Q = NeuralNetwork.load('{0:s}episode_{1:07d}.qnet'.format(self.path, self.__episode))
                    self.__T = NeuralNetwork.load('{0:s}episode_{1:07d}.tnet'.format(self.path, self.__episode))
                    self.__load_rlcd_file('{0:s}episode_{1:07d}.rlcd'.format(self.path, self.__episode))

                    self.__x_max = self.__episode if self.__x_max == self.__loaded_episode else self.__x_max
                    self.__loaded_episode = self.__episode
                    self.__plotParameters()
                    self.__plotQ()
                    self.__plotA()
                    self.__plotRewards()

                else:
                    print("Error unknown command")

        if self.__update_plot:
            self.__update_plot = False
            self.__fig.canvas.draw()
            # https://bastibe.de/2013-05-30-speeding-up-matplotlib.html
            # https://stackoverflow.com/questions/8955869/why-is-plotting-with-matplotlib-so-slow

        return True

    def __load_rlcd_file(self, file):
        def load_regularization(file):
            regularization = []
            for i in range(np.load(file)):
                regularization.append((str(np.load(file)), np.load(file)))
            return None if regularization == [] else regularization

        with open(file, 'rb') as file:
            self.__episodes = np.load(file)
            self.__ep_steps = np.load(file)
            _ = (np.load(file), np.load(file))
            self.__ep_return = np.load(file)
            self.__ep_tr_log = np.load(file)
            self.__R_max_size = int(np.load(file))
            self.__R_entries = int(np.load(file))
            _ = (np.load(file), np.load(file), np.load(file), np.load(file), np.load(file))
            self.__discount_factor = np.load(file)
            self.__update_factor = np.load(file)
            self.__replay_batch_size = np.load(file)
            _ = (np.load(file), np.load(file))
            _ = load_regularization(file)
            self.__Q_tr_freq = np.load(file)
            _ = (np.load(file), np.load(file))
            _ = load_regularization(file)
            self.__P_tr_freq = np.load(file)
            _ = (np.load(file),np.load(file), np.load(file), np.load(file))
            self.__tr_on_ep_prob = np.load(file)

    def __plotParameters(self):
        self.__axs_text.clear()
        self.__axs_text.set_xticks([])
        self.__axs_text.set_yticks([])
        self.__axs_text.set_navigate(False)
        self.__axs_text.axis('off')

        left_text, left_val, right_text, right_val = 0.0, 0.5, 0.55, 0.9
        self.__axs_text.text(left_text, 0.95, "Episode:")
        self.__axs_text.text(left_val, 0.95, "{0:d} / {1:d}".format(self.__episode, self.__episodes), ha='right')
        self.__axs_text.text(right_text, 0.95, "Steps:")
        self.__axs_text.text(right_val, 0.95, "{0:d} / {1:d}".format(self.__steps, self.__ep_steps), ha='right')
        self.__axs_text.text(left_text, 0.90, "Replay Buffer:")
        self.__axs_text.text(left_val, 0.90, "{0:d} / {1:d}".format(self.__R_entries, self.__R_max_size), ha='right')
        self.__axs_text.text(right_text, 0.90, "Replay Batch:")
        self.__axs_text.text(right_val, 0.90, "{0:d}".format(self.__replay_batch_size*self.__steps), ha='right')

        left_text, left_val, right_text, right_val = 0.0, 0.4, 0.5, 0.9
        self.__axs_text.text(left_text, 0.10, "Discount Factor:")
        self.__axs_text.text(left_val, 0.10, "{0:.3f}".format(self.__discount_factor), ha='right')
        self.__axs_text.text(right_text, 0.10, "Update Factor:")
        self.__axs_text.text(right_val, 0.10, "{0:.3f}".format(self.__update_factor), ha='right')
        self.__axs_text.text(left_text, 0.05, "Action Noise:")
        self.__axs_text.text(left_val, 0.05, "{0:.3f}".format(self.__act_noise), ha='right')
        self.__axs_text.text(right_text, 0.05, "On-Episode Training:")
        self.__axs_text.text(right_val, 0.05, "{0:.3f}".format(self.__tr_on_ep_prob), ha='right')
        self.__axs_text.text(left_text, 0.00, "Q Training Frequency:")
        self.__axs_text.text(left_val, 0.00, "{0:d}".format(self.__Q_tr_freq), ha='right')
        self.__axs_text.text(right_text, 0.00, "P Training Frequency:")
        self.__axs_text.text(right_val, 0.00, "{0:d}".format(self.__P_tr_freq), ha='right')

        self.__update_plot = True

    def __plotTrajectory(self, destination, trajectory):
        self.__axs_T.clear()
        self.__axs_T.set_xlim(-2.5, 2.5)
        self.__axs_T.set_ylim(-1.5, 1.5)
        self.__axs_T.set_xticks(np.linspace(-1.5, 1.5, num=7, endpoint=True))
        self.__axs_T.set_yticks(np.linspace(-1.5, 1.5, num=7, endpoint=True))
        self.__axs_T.set_title('Last Trajectory')
        self.__axs_T.add_patch(plt.Circle((0, 0), 0.1, color='orange', fill=False))
        self.__axs_T.add_patch(plt.Circle((0, 0), 1.5, color='orange', fill=False))
        if (len(destination) <= 1) or (len(trajectory) <= 1): return;
        self.__axs_T.scatter(destination[0], destination[1], color='blue', alpha=0.8, linewidth=2, linestyle='solid')
        self.__axs_T.scatter(trajectory[0,0], trajectory[0,1], color='black', alpha=0.8, linewidth=2, linestyle='solid')
        self.__axs_T.plot(trajectory[:,0], trajectory[:,1], color='black', alpha=0.8, linewidth=2, linestyle='solid')

        self.__update_plot = True

    def __plotQ(self):
        axis_1D, axis_2D = self.__axis_1D, self.__axis_2D

        # Plot predicted cummulative reward
        var1, var2, _ = self.__get_axes_variables()
        obs_aux, act_aux = np.array([self.__obs, ] * len(axis_2D)), np.array([self.__act, ] * len(axis_2D))

        if var1<self.__obs_size: obs_aux[:, var1] = axis_2D[:,0]
        else:                    act_aux[:, var1-self.__obs_size] = axis_2D[:,0]
        if var2<self.__obs_size: obs_aux[:, var2] = axis_2D[:,1]
        else:                    act_aux[:, var2-self.__obs_size] = axis_2D[:,1]

        Q = np.array(self.__Q.compute([obs_aux, act_aux])).reshape((self.resolution, self.resolution))

        self.__axs_Q.clear()
        #self.__axs_Q.clabel(self.__axs_Q.contour3D(axis_1D, axis_1D, Q, 50, cmap='binary'), fontsize=9, inline=1)
        self.__axs_Q.plot_surface(self.__mx, self.__my, Q, cmap=self.__colormap)
        self.__axs_Q.invert_yaxis()
        self.__axs_Q.set_xlabel('S{0:d}'.format(var1) if var1<self.__obs_size else 'A{0:d}'.format(var1-self.__obs_size))
        self.__axs_Q.set_ylabel('S{0:d}'.format(var2) if var2<self.__obs_size else 'A{0:d}'.format(var2-self.__obs_size))
        self.__axs_Q.set_title('Q Function')

        self.__update_plot = True

    def __plotA(self):
        axis_1D, axis_2D = self.__axis_1D, self.__axis_2D

        #Plot policy
        var1, var2, var3 = self.__get_axes_variables()
        if (var1 >= self.__obs_size) or (var2 >= self.__obs_size): return
        obs_aux = np.array([self.__obs, ] * len(axis_2D))

        obs_aux[:,var1] = axis_2D[:,0]
        obs_aux[:,var2] = axis_2D[:,1]

        self.__axs_A.clear()
        self.__axs_A.invert_yaxis()
        self.__axs_A.set_xlabel('S{0:d}'.format(var1))
        self.__axs_A.set_ylabel('S{0:d}'.format(var2))

        if var3 >= 0:
            self.__axs_A.set_title('Action[{0:d}]'.format(var3))
            A = np.array(self.__P.compute(obs_aux)).reshape((self.resolution, self.resolution, -1))
            self.__axs_A.plot_surface(self.__mx, self.__my, A[:, :, var3], cmap=self.__colormap)
            #self.__axs_A.clabel(self.__axs_A.contour3D(axis_1D, axis_1D, A[:,:,var3], 50, cmap='binary'), fontsize=9, inline=1)
        else:
            self.__axs_A.set_title('Target')
            T = np.array(self.__T.compute(obs_aux)).reshape((self.resolution, self.resolution))
            self.__axs_A.plot_surface(self.__mx, self.__my, T, cmap=self.__colormap)

        self.__update_plot = True

    def __plotRewards(self):
        if self.__loaded_episode == 0: return

        # Get plot parameters
        x_min, x_max = self.__x_min, self.__x_max if self.__x_max > self.__x_min else self.__loaded_episode
        N = 100 if x_max-x_min > 100 else x_max-x_min
        p_low, p_high = ((1,99), (5,95), (10,90), (25,75))[self.__percentile_idx]
        episodes = np.linspace(x_min, x_max, num=N, endpoint=True)

        # First plot
        self.__axs_R[0].clear()
        self.__axs_R[0].set_title('Q function Training Loss')
        # Calculate and plot data
        data, color = self.__ep_tr_log[x_min:x_max, 0].reshape(N,-1), 'black'
        data_mean = np.mean(data, axis=1)
        data_min, data_low, data_high, data_max = np.percentile(data, [0, p_low, p_high, 100], axis=1)
        self.__axs_R[0].plot(episodes, data_mean, color=color, alpha=0.8, linewidth=2, linestyle='solid')
        self.__axs_R[0].fill_between(episodes, data_low, data_high, color=color, alpha=0.1)
        if self.__max_min: self.__axs_R[0].plot(episodes, data_max, color=color, alpha=0.3, linewidth=1, linestyle='--')
        if self.__max_min: self.__axs_R[0].plot(episodes, data_min, color=color, alpha=0.3, linewidth=1, linestyle='--')
        # Configure plot axes
        if not self.__max_min: data_min, data_max = np.min([data_low, data_mean]), np.max([data_high, data_mean])
        digits = int(np.floor(np.log10(np.max(np.abs([data_min, data_max])))) - 1)
        y_min = round(np.min(data_min) - 0.5 * pow(10, digits), -digits)
        y_max = round(np.max(data_max) + 0.5 * pow(10, digits), -digits)
        self.__axs_R[0].set_xlim(x_min, x_max)
        self.__axs_R[0].set_ylim(y_min, y_max)
        self.__axs_R[0].set_yticks(np.linspace(y_min, y_max, num=6, endpoint=True))

        # Second plot
        self.__axs_R[1].clear()
        self.__axs_R[1].set_title('Policy Training Loss')
        # Calculate and plot data
        data, color = self.__ep_tr_log[x_min:x_max, 1].reshape(N,-1), 'black'
        data_mean = np.mean(data, axis=1)
        data_min, data_low, data_high, data_max = np.percentile(data, [0, p_low, p_high, 100], axis=1)
        self.__axs_R[1].plot(episodes, data_mean, color=color, alpha=0.8, linewidth=2, linestyle='solid')
        self.__axs_R[1].fill_between(episodes, data_low, data_high, color=color, alpha=0.1)
        if self.__max_min: self.__axs_R[1].plot(episodes, data_max, color=color, alpha=0.3, linewidth=1, linestyle='--')
        if self.__max_min: self.__axs_R[1].plot(episodes, data_min, color=color, alpha=0.3, linewidth=1, linestyle='--')
        # Configure plot axes
        if not self.__max_min: data_min, data_max = np.min([data_low, data_mean]), np.max([data_high, data_mean])
        digits = int(np.floor(np.log10(np.max(np.abs([data_min, data_max])))) - 1)
        y_min = round(np.min(data_min) - 0.5 * pow(10, digits), -digits)
        y_max = round(np.max(data_max) + 0.5 * pow(10, digits), -digits)
        self.__axs_R[1].set_xlim(x_min, x_max)
        self.__axs_R[1].set_ylim(y_min, y_max)
        self.__axs_R[1].set_yticks(np.linspace(y_min, y_max, num=6, endpoint=True))

        # Third plot
        self.__axs_R[2].clear()
        self.__axs_R[2].set_title('Infinite-Horizon Discounted Return\nReal(red) vs. Predicted(blue)')
        # Calculate and plot data
        data, color = self.__ep_return[x_min:x_max, 0].reshape(N,-1), 'red'
        data_mean = np.mean(data, axis=1)
        data_min, data_low, data_high, data_max = np.percentile(data, [0, p_low, p_high, 100], axis=1)
        self.__axs_R[2].plot(episodes, data_mean, color=color, alpha=0.8, linewidth=2, linestyle='solid')
        self.__axs_R[2].fill_between(episodes, data_low, data_high, color=color, alpha=0.1)
        if self.__max_min: self.__axs_R[2].plot(episodes, data_max, color=color, alpha=0.3, linewidth=1, linestyle='--')
        if self.__max_min: self.__axs_R[2].plot(episodes, data_min, color=color, alpha=0.3, linewidth=1, linestyle='--')
        # Calculate and plot data
        data, color = self.__ep_return[x_min:x_max, 1].reshape(N,-1), 'blue'
        data_mean2 = np.mean(data, axis=1)
        data_min2, data_low2, data_high2, data_max2 = np.percentile(data, [0, p_low, p_high, 100], axis=1)
        self.__axs_R[2].plot(episodes, data_mean2, color=color, alpha=0.8, linewidth=2, linestyle='solid')
        self.__axs_R[2].fill_between(episodes, data_low2, data_high2, color=color, alpha=0.1)
        if self.__max_min: self.__axs_R[2].plot(episodes, data_max2, color=color, alpha=0.3, linewidth=1, linestyle='--')
        if self.__max_min: self.__axs_R[2].plot(episodes, data_min2, color=color, alpha=0.3, linewidth=1, linestyle='--')
        # Configure plot axes
        data_min = np.min([data_min, data_min2] if self.__max_min else [data_low, data_low2, data_mean, data_mean2])
        data_max = np.max([data_max, data_max2] if self.__max_min else [data_high, data_high2, data_mean, data_mean2])
        digits = int(np.floor(np.log10(np.max(np.abs([data_min, data_max])))) - 1)
        y_min = round(np.min(data_min) - 0.5 * pow(10, digits), -digits)
        y_max = round(np.max(data_max) + 0.5 * pow(10, digits), -digits)
        self.__axs_R[2].set_xlim(x_min, x_max)
        self.__axs_R[2].set_ylim(y_min, y_max)
        self.__axs_R[2].set_yticks(np.linspace(y_min, y_max, num=6, endpoint=True))

        # Fourth plot
        self.__axs_R[3].clear()
        self.__axs_R[3].set_title('Infinite-Horizon Discounted Return\n Root Mean Squared Error')
        # Calculate and plot data
        data, color = self.__ep_return[x_min:x_max, 2].reshape(N,-1), 'purple'
        data_mean = np.mean(data, axis=1)
        data_min, data_low, data_high, data_max = np.percentile(data, [0, p_low, p_high, 100], axis=1)
        self.__axs_R[3].plot(episodes, data_mean, color=color, alpha=0.8, linewidth=2, linestyle='solid')
        self.__axs_R[3].fill_between(episodes, data_low, data_high, color=color, alpha=0.1)
        if self.__max_min: self.__axs_R[3].plot(episodes, data_max, color=color, alpha=0.3, linewidth=1, linestyle='--')
        if self.__max_min: self.__axs_R[3].plot(episodes, data_min, color=color, alpha=0.3, linewidth=1, linestyle='--')
        # Configure plot axes
        if not self.__max_min: data_min, data_max = np.min([data_low, data_mean]), np.max([data_high, data_mean])
        digits = int(np.floor(np.log10(np.max(np.abs([data_min, data_max])))) - 1)
        y_min = round(np.min(data_min) - 0.5 * pow(10, digits), -digits)
        y_max = round(np.max(data_max) + 0.5 * pow(10, digits), -digits)
        self.__axs_R[3].set_xlim(x_min, x_max)
        self.__axs_R[3].set_ylim(y_min, y_max)
        self.__axs_R[3].set_yticks(np.linspace(y_min, y_max, num=6, endpoint=True))

        # Request plot update
        self.__update_plot = True



if __name__ == '__main__':
    resolution = 50
    figsize = (14, 5)
    view = (15, 45)
    plots = [['Env-P', 'Env-P-diff', 'Env-Q', 'Env-V'],
            ['Simple-P', 'Simple-Q', 'Simple-V'],
            ['Complex-M', 'Complex-S', 'Complex-Q', 'Complex-V']]
    plot = plots[1][2]
    title = "Comparison "+plot
    action = 0 # 7 for complex agent

    P = [NeuralNetwork.load('{0:1d}.pnet'.format(1)),
         NeuralNetwork.load('{0:1d}.pnet'.format(2)),
         NeuralNetwork.load('{0:1d}.pnet'.format(3))]

    Q = [NeuralNetwork.load('{0:1d}.qnet'.format(1)),
         NeuralNetwork.load('{0:1d}.qnet'.format(2)),
         NeuralNetwork.load('{0:1d}.qnet'.format(3))]

    obs = np.zeros(P[0].input_layers[0].out_shape)
    act = np.zeros(P[0].output_layers[0].out_shape)

    colormap = [
        mpl.colors.ListedColormap(mpl.cm.bone(np.linspace(0, 1, 1000))[150:850, :-1]),
        mpl.colors.ListedColormap(mpl.cm.winter(np.linspace(0, 1, 1000))[0:1000, :-1] * 0.9),
        mpl.colors.ListedColormap(mpl.cm.gist_heat(np.linspace(0, 1, 1000))[0:900, :-1]),
        mpl.colors.ListedColormap(mpl.cm.summer(np.linspace(0, 1, 1000))[0:800, :-1])
    ][1]

    fig = plt.figure(figsize=figsize)
    fig.canvas.set_window_title(title)
    if plot == plots[0][0]:
        ax1 = fig.add_subplot(1, 4, 1, projection='3d')
        axs = [ax1,
           fig.add_subplot(1, 4, 2, projection='3d', sharez=ax1),
           fig.add_subplot(1, 4, 3, projection='3d', sharez=ax1),
           fig.add_subplot(1, 4, 4, projection='3d', sharez=ax1)]
        axs[3].clear()
        axs[3].set_xlabel("X")
        axs[3].set_ylabel("Y")
        axs[3].view_init(*view)
    else:
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        axs = [ax1,
               fig.add_subplot(1, 3, 2, projection='3d', sharez=ax1),
               fig.add_subplot(1, 3, 3, projection='3d', sharez=ax1)]
    axs[0].clear()
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[1].clear()
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[2].clear()
    axs[2].set_xlabel("X")
    axs[2].set_ylabel("Y")
#    axs[0].invert_yaxis()
#    axs[1].invert_yaxis()
#    axs[2].invert_yaxis()
    axs[0].view_init(*view)
    axs[1].view_init(*view)
    axs[2].view_init(*view)

    axis_1D = np.linspace(-1, 1, num=resolution)
    mx, my = np.meshgrid(axis_1D, axis_1D)
    mx2 = np.array(mx).reshape(-1, 1)
    my2 = np.array(my).reshape(-1, 1)
    axis_2D = np.concatenate((mx2, my2), axis=1)

    # Plot predicted cummulative reward
    obs_aux, act_aux = np.array([obs, ] * len(axis_2D)), np.array([act, ] * len(axis_2D))
    obs_aux[:, 0] = axis_2D[:, 0]
    obs_aux[:, 1] = axis_2D[:, 1]


    if plot == plots[0][0]:
        fig.suptitle("First Environment's Policy", fontsize=16)
        axs[0].plot_surface(mx, my, np.arctan2(-my, -mx).reshape((resolution, resolution)) / np.pi, cmap=colormap)
        axs[0].set_title('arctan2(-y, -x)')
        axs[1].plot_surface(mx, my, P[0].compute([obs_aux, act_aux]).reshape((resolution, resolution, -1))[:,:,0], cmap=colormap)
        axs[1].set_title('DDPG')
        axs[2].plot_surface(mx, my, P[1].compute([obs_aux, act_aux]).reshape((resolution, resolution, -1))[:,:,0], cmap=colormap)
        axs[2].set_title('TD3')
        P[2].compute([obs_aux, act_aux])
        axs[3].plot_surface(mx, my, P[2].output_layers[0].input_layers[0].a.reshape((resolution, resolution, -1))[:,:,0], cmap=colormap)
        axs[3].set_title('SAC')
        results = [P[0].output_layers[0].a, P[1].output_layers[0].a, P[2].output_layers[0].input_layers[0].a]
        axs[0].set_zlim3d(np.min(results), np.max(results))
    elif plot == plots[0][1]:
        fig.suptitle("First Environment's Policy Absolute Error", fontsize=16)
        real = np.arctan2(-my, -mx).reshape((resolution, resolution)) / np.pi
        P[2].compute([obs_aux, act_aux])
        results = [P[0].compute([obs_aux, act_aux]).reshape((resolution, resolution, -1))[:,:,0],
                   P[1].compute([obs_aux, act_aux]).reshape((resolution, resolution, -1))[:, :, 0],
                   P[2].output_layers[0].input_layers[0].a.reshape((resolution, resolution, -1))[:, :, 0]]
        results = [np.min([np.abs(results[0]-real), np.abs(results[0]-real+2), np.abs(results[0]-real-2)], axis=0),
                   np.min([np.abs(results[1]-real), np.abs(results[1]-real+2), np.abs(results[1]-real-2)], axis=0),
                   np.min([np.abs(results[2]-real), np.abs(results[2]-real+2), np.abs(results[2]-real-2)], axis=0)]
        axs[0].plot_surface(mx, my, results[0], cmap=colormap)
        axs[0].set_title('DDPG')
        axs[1].plot_surface(mx, my, results[1], cmap=colormap)
        axs[1].set_title('TD3')
        axs[2].plot_surface(mx, my, results[2], cmap=colormap)
        axs[2].set_title('SAC')
        axs[0].set_zlim3d(np.min(results), np.max(results))
    elif plot == plots[0][2]:
        fig.suptitle("First Environment's State-Action Value Function", fontsize=16)
        axs[0].plot_surface(mx, my, Q[0].compute([obs_aux, act_aux]).reshape((resolution, resolution)), cmap=colormap)
        axs[0].set_title('DDPG')
        axs[1].plot_surface(mx, my, Q[1].compute([obs_aux, act_aux]).reshape((resolution, resolution)), cmap=colormap)
        axs[1].set_title('TD3')
        axs[2].plot_surface(mx, my, Q[2].compute([obs_aux, act_aux]).reshape((resolution, resolution)), cmap=colormap)
        axs[2].set_title('SAC')
        results = [Q[0].output_layers[0].a, Q[1].output_layers[0].a, Q[2].output_layers[0].a]
        axs[0].set_zlim3d(np.min(results), np.max(results))
    elif plot == plots[0][3]:
        fig.suptitle("First Environment's State Value Function", fontsize=16)
        axs[0].plot_surface(mx, my, Q[0].compute([obs_aux, P[0].compute([obs_aux, act_aux])]).reshape((resolution, resolution)), cmap=colormap)
        axs[0].set_title('DDPG')
        axs[1].plot_surface(mx, my, Q[1].compute([obs_aux, P[1].compute([obs_aux, act_aux])]).reshape((resolution, resolution)), cmap=colormap)
        axs[1].set_title('TD3')
        P[2].compute([obs_aux, act_aux])
        axs[2].plot_surface(mx, my, Q[2].compute([obs_aux, P[2].output_layers[0].input_layers[0].a]).reshape((resolution, resolution)), cmap=colormap)
        axs[2].set_title('SAC')
        results = [Q[0].output_layers[0].a, Q[1].output_layers[0].a, Q[2].output_layers[0].a]
        axs[0].set_zlim3d(np.min(results), np.max(results))
    elif plot == plots[1][0]:
        fig.suptitle("Second Environment's Policy", fontsize=16)
        axs[0].plot_surface(mx, my, P[0].compute([obs_aux, act_aux]).reshape((resolution, resolution, -1))[:,:,0], cmap=colormap)
        axs[0].set_title('DDPG')
        axs[1].plot_surface(mx, my, P[1].compute([obs_aux, act_aux]).reshape((resolution, resolution, -1))[:,:,0], cmap=colormap)
        axs[1].set_title('TD3')
        P[2].compute([obs_aux, act_aux])
        axs[2].plot_surface(mx, my, P[2].output_layers[0].input_layers[0].a.reshape((resolution, resolution, -1))[:,:,action], cmap=colormap)
        axs[2].set_title('SAC')
        results = [P[0].output_layers[0].a, P[1].output_layers[0].a, P[2].output_layers[0].input_layers[0].a]
        axs[0].set_zlim3d(np.min(results), np.max(results))
    elif plot == plots[1][1]:
        fig.suptitle("Second Environment's State-Action Value Function", fontsize=16)
        axs[0].plot_surface(mx, my, Q[0].compute([obs_aux, act_aux]).reshape((resolution, resolution)), cmap=colormap)
        axs[0].set_title('DDPG')
        axs[1].plot_surface(mx, my, Q[1].compute([obs_aux, act_aux]).reshape((resolution, resolution)), cmap=colormap)
        axs[1].set_title('TD3')
        axs[2].plot_surface(mx, my, Q[2].compute([obs_aux, act_aux]).reshape((resolution, resolution)), cmap=colormap)
        axs[2].set_title('SAC')
        results = [Q[0].output_layers[0].a, Q[1].output_layers[0].a, Q[2].output_layers[0].a]
        axs[0].set_zlim3d(np.min(results), np.max(results))
    elif plot == plots[1][2]:
        fig.suptitle("Third Environment's State Value Function", fontsize=16)
        axs[0].plot_surface(mx, my, Q[0].compute([obs_aux, P[0].compute([obs_aux, act_aux])]).reshape((resolution, resolution)), cmap=colormap)
        axs[0].set_title('DDPG')
        axs[1].plot_surface(mx, my, Q[1].compute([obs_aux, P[1].compute([obs_aux, act_aux])]).reshape((resolution, resolution)), cmap=colormap)
        axs[1].set_title('TD3')
        P[2].compute([obs_aux, act_aux])
        axs[2].plot_surface(mx, my, Q[0].compute([obs_aux, P[2].output_layers[0].input_layers[0].a]).reshape((resolution, resolution)), cmap=colormap)
        axs[2].set_title('SAC')
        results = [Q[0].output_layers[0].a, Q[1].output_layers[0].a, Q[0].output_layers[0].a]
        axs[0].set_zlim3d(np.min(results), np.max(results))
    elif plot == plots[2][0]:
        fig.suptitle("Third Environment's Policy Mean", fontsize=16)
        P[0].compute([obs_aux, act_aux])
        axs[0].plot_surface(mx, my, P[0].output_layers[0].input_layers[0].a.reshape((resolution, resolution, -1))[:,:,action], cmap=colormap)
        axs[0].set_title('alpha = 0.05')
        P[1].compute([obs_aux, act_aux])
        axs[1].plot_surface(mx, my, P[1].output_layers[0].input_layers[0].a.reshape((resolution, resolution, -1))[:,:,action], cmap=colormap)
        axs[1].set_title('alpha = 0.01')
        P[2].compute([obs_aux, act_aux])
        axs[2].plot_surface(mx, my, P[2].output_layers[0].input_layers[0].a.reshape((resolution, resolution, -1))[:,:,action], cmap=colormap)
        axs[2].set_title('alpha = 0.005')
        results = [P[0].output_layers[0].input_layers[0].a, P[1].output_layers[0].input_layers[0].a, P[2].output_layers[0].input_layers[0].a]
        axs[0].set_zlim3d(np.min(results), np.max(results))
    elif plot == plots[2][1]:
        fig.suptitle("Third Environment's Policy Standard Deviation", fontsize=16)
        P[0].compute([obs_aux, act_aux])
        axs[0].plot_surface(mx, my, P[0].output_layers[0].input_layers[1].a.reshape((resolution, resolution, -1))[:,:,action], cmap=colormap)
        axs[0].set_title('alpha = 0.05')
        P[1].compute([obs_aux, act_aux])
        axs[1].plot_surface(mx, my, P[1].output_layers[0].input_layers[1].a.reshape((resolution, resolution, -1))[:,:,action], cmap=colormap)
        axs[1].set_title('alpha = 0.01')
        P[2].compute([obs_aux, act_aux])
        axs[2].plot_surface(mx, my, P[2].output_layers[0].input_layers[1].a.reshape((resolution, resolution, -1))[:,:,action], cmap=colormap)
        axs[2].set_title('alpha = 0.005')
        results = [P[0].output_layers[0].input_layers[1].a, P[1].output_layers[0].input_layers[1].a, P[2].output_layers[0].input_layers[1].a]
        axs[0].set_zlim3d(np.min(results), np.max(results))
    elif plot == plots[2][2]:
        fig.suptitle("Third Environment's State-Action Value Function", fontsize=16)
        axs[0].plot_surface(mx, my, Q[0].compute([obs_aux, act_aux]).reshape((resolution, resolution)), cmap=colormap)
        axs[0].set_title('alpha = 0.05')
        axs[1].plot_surface(mx, my, Q[1].compute([obs_aux, act_aux]).reshape((resolution, resolution)), cmap=colormap)
        axs[1].set_title('alpha = 0.01')
        axs[2].plot_surface(mx, my, Q[2].compute([obs_aux, act_aux]).reshape((resolution, resolution)), cmap=colormap)
        axs[2].set_title('alpha = 0.005')
        results = [Q[0].output_layers[0].a, Q[1].output_layers[0].a, Q[2].output_layers[0].a]
        axs[0].set_zlim3d(np.min(results), np.max(results))
    elif plot == plots[2][3]:
        fig.suptitle("Third Environment's State Value Function", fontsize=16)
        P[0].compute([obs_aux, act_aux])
        axs[0].plot_surface(mx, my, Q[0].compute([obs_aux, P[0].output_layers[0].input_layers[0].a]).reshape((resolution, resolution)), cmap=colormap)
        axs[0].set_title('alpha = 0.05')
        P[1].compute([obs_aux, act_aux])
        axs[1].plot_surface(mx, my, Q[1].compute([obs_aux, P[1].output_layers[0].input_layers[0].a]).reshape((resolution, resolution)), cmap=colormap)
        axs[1].set_title('alpha = 0.01')
        P[2].compute([obs_aux, act_aux])
        axs[2].plot_surface(mx, my, Q[2].compute([obs_aux, P[2].output_layers[0].input_layers[0].a]).reshape((resolution, resolution)), cmap=colormap)
        axs[2].set_title('alpha = 0.005')
        results = [Q[0].output_layers[0].a, Q[1].output_layers[0].a, Q[2].output_layers[0].a]
        axs[0].set_zlim3d(np.min(results), np.max(results))

    plt.show()
