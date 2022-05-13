% Script to compute DECODED's attention and motor imagery PI's
% To use, load the sessions to be used on the classification
clear all
close all
%%%%%% Configuration variables
initial_path = pwd();
read_dir = 'ReadFiles';
save_dir = 'Results';
% Basic
plot_results = 1; % 1 or 0
shift = 100;
n_electrodes = 31; % Number of electrodes, with the ones of EOG included
fm = 200;
electrodes_selected =  {'F3', 'FZ', 'FC1','FCZ','C1','CZ','CP1','CPZ', 'FC5', 'FC3','C5','C3','CP5','CP3','P3','PZ','F4','FC2','FC4','FC6','C2','C4','CP2','CP4','C6','CP6','P4'};

% CSP
m = 4;
relax_class = [402];
MI_class = [404, 752, 753, 754];
count_class = [406, 756, 757, 758];

% Processing
processing_windows = 300;


%% Preprocessing: band-pass filters
BP_conf(1).order=2;
BP_conf(1).low_frequency=5;
BP_conf(1).high_frequency=10;
% filter_parameters= Filter parameters: A,B,C,D  and Xnn parameters of the state-space filter
BP_conf(1).filter_parameters = band_pass_filter_setup(fm, n_electrodes, BP_conf(1).order, BP_conf(1).low_frequency, BP_conf(1).high_frequency);

BP_conf(2).order=2;
BP_conf(2).low_frequency=10;
BP_conf(2).high_frequency=15;
% filter_parameters= Filter parameters: A,B,C,D  and Xnn parameters of the state-space filter
BP_conf(2).filter_parameters = band_pass_filter_setup(fm, n_electrodes, BP_conf(2).order, BP_conf(2).low_frequency, BP_conf(2).high_frequency);

BP_conf(3).order=2;
BP_conf(3).low_frequency=15;
BP_conf(3).high_frequency=20;
% filter_parameters= Filter parameters: A,B,C,D  and Xnn parameters of the state-space filter
BP_conf(3).filter_parameters = band_pass_filter_setup(fm, n_electrodes, BP_conf(3).order, BP_conf(3).low_frequency, BP_conf(3).high_frequency);

BP_conf(4).order=2;
BP_conf(4).low_frequency=20;
BP_conf(4).high_frequency=25;
% filter_parameters= Filter parameters: A,B,C,D  and Xnn parameters of the state-space filter
BP_conf(4).filter_parameters = band_pass_filter_setup(fm, n_electrodes, BP_conf(4).order, BP_conf(4).low_frequency, BP_conf(4).high_frequency);

BP_conf(5).order=2;
BP_conf(5).low_frequency=30;
BP_conf(5).high_frequency=35;
% filter_parameters= Filter parameters: A,B,C,D  and Xnn parameters of the state-space filter
BP_conf(5).filter_parameters = band_pass_filter_setup(fm, n_electrodes, BP_conf(5).order, BP_conf(5).low_frequency, BP_conf(5).high_frequency);

BP_conf(6).order=2;
BP_conf(6).low_frequency=35;
BP_conf(6).high_frequency=45;
% filter_parameters= Filter parameters: A,B,C,D  and Xnn parameters of the state-space filter
BP_conf(6).filter_parameters = band_pass_filter_setup(fm, n_electrodes, BP_conf(6).order, BP_conf(6).low_frequency, BP_conf(6).high_frequency);

BP_conf(7).order=2;
BP_conf(7).low_frequency=55;
BP_conf(7).high_frequency=60;
% filter_parameters= Filter parameters: A,B,C,D  and Xnn parameters of the state-space filter
BP_conf(7).filter_parameters = band_pass_filter_setup(fm, n_electrodes, BP_conf(7).order, BP_conf(7).low_frequency, BP_conf(7).high_frequency);

BP_conf(8).order=2;
BP_conf(8).low_frequency=60;
BP_conf(8).high_frequency=70;
% filter_parameters= Filter parameters: A,B,C,D  and Xnn parameters of the state-space filter
BP_conf(8).filter_parameters = band_pass_filter_setup(fm, n_electrodes, BP_conf(8).order, BP_conf(8).low_frequency, BP_conf(8).high_frequency);

BP_conf(9).order=2;
BP_conf(9).low_frequency=70;
BP_conf(9).high_frequency=80;
% filter_parameters= Filter parameters: A,B,C,D  and Xnn parameters of the state-space filter
BP_conf(9).filter_parameters = band_pass_filter_setup(fm, n_electrodes, BP_conf(9).order, BP_conf(9).low_frequency, BP_conf(9).high_frequency);

BP_conf(10).order=2;
BP_conf(10).low_frequency=80;
BP_conf(10).high_frequency=90;
% filter_parameters= Filter parameters: A,B,C,D  and Xnn parameters of the state-space filter
BP_conf(10).filter_parameters = band_pass_filter_setup(fm, n_electrodes, BP_conf(10).order, BP_conf(10).low_frequency, BP_conf(10).high_frequency);

%% Read files
[session_files, session_path] = uigetfile('.\Registers\CSV\*.csv', 'MultiSelect', 'on');

for k = 1:length(session_files)
    vector_session(k) = load_session(fullfile(session_path, session_files{k}));
end

channels = match_electrodes(vector_session(1).electrodes, electrodes_selected);

%% Preprocessing (apply band-pass filter)
for k = 1:length(vector_session)
    % Iterate every band-pass filter
    vector_session(k).data_preprocessed_EEG = vector_session(k).data_EEG;
    for i = 1:length(BP_conf)
        % Iterate over every epoch
        for j=1:floor(length(vector_session(k).data_EEG) / shift)           
            [vector_session(k).data_preprocessed_EEG(i*size(vector_session(k).data_EEG, 1)+1:(i+1)*size(vector_session(k).data_EEG, 1), (j-1)*shift+1:j*shift), new_Xnn] = band_pass_filter_multiple(vector_session(k).data_EEG(:, (j-1)*shift+1:j*shift), BP_conf(i));
            BP_conf(i).filter_parameters.Xnn = new_Xnn;
        end
    end
    disp(['Trial ' int2str(k) ' filtered...']);
end

%% Processing and first classification (compute CSP classification)
% 4 different classificators
% 1: Alpha and beta, MI vs Relax
% 2: Alpha and beta, MI vs Count
% 3: Gamma, MI  vs Relax
% 4: Gamma, MI vs count
CSP_classify = cell(4, length(vector_session));
for n_CSP=1:4
    n_rows = size(vector_session(1).data_preprocessed_EEG,1); %number of rows of data_preprocessed, same for all trials and classes
    n_ch = n_electrodes;
    
    if ismember(n_CSP, [1, 2])
        fbands = 4;
    else
        fbands = 6;
    end
    
    data_previous_trials_fbands_class1 = cell(fbands,length(vector_session));
    data_previous_trials_fbands_class2 = cell(fbands,length(vector_session));
    
    %% Prepare data for each trial
    for j=1:(length(vector_session))
        %%% Prepare epochs for training and test
        epochs_training{j} = [];
        epochs_valid{j} = [];
        tasks_epoch{j} = zeros(1,floor(length(vector_session(j).task_EEG)/shift));
        tasks{j} = zeros(1,floor(length(vector_session(j).task_EEG)/shift));

        for z=1:floor(length(vector_session(j).task_EEG)/shift) %how many epochs
            tasks_epoch{j}(z)=mode(vector_session(j).task_EEG((z-1)*shift+1:z*shift)); %computes the mode through each epoch to get one task per epoch
            epochs_valid{j} = [epochs_valid{j} z];
            % Change labels for MI and att
            if ismember(n_CSP, [1, 3]) % (MI + Count) vs relax
                if ismember(tasks_epoch{j}(z), [403, 404, 751, 752, 753, 754, 761]) % MI % Cambiar: añadir aquí también las task de slopes
                    tasks{j}(z)= 1;
                elseif ismember(tasks_epoch{j}(z), [405, 406, 755, 756, 757, 758, 759]) % Count
                    tasks{j}(z) = 1;
                elseif ismember(tasks_epoch{j}(z), [400, 401, 402]) % Relax
                    tasks{j}(z)= 0;
                end

                % Epochs valid for training (not all task used for training)
                if ismember(tasks_epoch{j}(z), [402, 404, 752, 754])
                    epochs_training{j} = [epochs_training{j} z];
                end
            else % MI vs Count
                if ismember(tasks_epoch{j}(z), [403, 404, 751, 752, 753, 754, 761]) % MI % Cambiar: añadir aquí también las task de slopes
                    tasks{j}(z)= 1;
                elseif ismember(tasks_epoch{j}(z), [405, 406, 755, 756, 757, 758, 759]) % Count
                    tasks{j}(z) = 0;
                elseif ismember(tasks_epoch{j}(z), [400, 401, 402])% Relax
                    tasks{j}(z)= 0;
                end

                % Epochs valid for training (not all task used for training)
                if ismember(tasks_epoch{j}(z), [404, 406, 752, 754, 756, 758])
                    epochs_training{j} = [epochs_training{j} z];
                end
            end
        end
        
        %%% Separe frequency band filters
        if ismember(n_CSP, [1, 3]) % (MI + Count) vs Relax
            class_1 = [relax_class];
            class_2 = [MI_class];
        else % MI vs Count
            class_1 = [count_class];
            class_2 = [MI_class];
        end
        i_class1 = find(ismember(vector_session(j).task_EEG, class_1));
        i_class2 = find(ismember(vector_session(j).task_EEG, class_2));
        
        %% FBCSP
        % Divide in frequency bands
        for f=1:fbands
            if ismember(n_CSP, [1, 2]) % Alpha band (filters 1 to 4)
                % Get the data from each bandpass filter
                data_previous_trials_fbands_class1{f,j}=vector_session(j).data_preprocessed_EEG(n_ch+(f-1)*n_ch+1:n_ch+f*n_ch,i_class1);
                data_previous_trials_fbands_class2{f,j}=vector_session(j).data_preprocessed_EEG(n_ch+(f-1)*n_ch+1:n_ch+f*n_ch,i_class2);

                % Get the from only channels selected
                data_previous_trials_fbands_class1{f,j} = data_previous_trials_fbands_class1{f,j}(channels,:);
                data_previous_trials_fbands_class2{f,j} = data_previous_trials_fbands_class2{f,j}(channels,:);
            else % Gamma band (filters 5 to 10)
                % Get the data from each bandpass filter
                data_previous_trials_fbands_class1{f,j}=vector_session(j).data_preprocessed_EEG(n_ch+(f-1+4)*n_ch+1:n_ch+(f+4)*n_ch,i_class1);
                data_previous_trials_fbands_class2{f,j}=vector_session(j).data_preprocessed_EEG(n_ch+(f-1+4)*n_ch+1:n_ch+(f+4)*n_ch,i_class2);

                % Get the from only channels selected
                data_previous_trials_fbands_class1{f,j} = data_previous_trials_fbands_class1{f,j}(channels,:);
                data_previous_trials_fbands_class2{f,j} = data_previous_trials_fbands_class2{f,j}(channels,:);
            end
        end  
    end
    
    %% CSP Cross-validation (leave-one-out)  
    W_CSP = cell(length(vector_session), fbands); % CSP spatial filters, one per trial (per fold of cross-validation)
    
    for k=1:length(vector_session) % k: nº of fold
        %% Compute CSP filter
        for f=1: fbands
            X1= data_previous_trials_fbands_class1(f,:);
            X2= data_previous_trials_fbands_class2(f,:);

            % Do not include trials for testing
            X1(k)=[]; %do not consider test trials for W spatial filter computation
            X2(k)=[];

            %Compute transformation matrix
            W=csp(X1,X2);
            W_CSP{k,f}=W;
        end
        % END Compute CSP filter
        %% Apply spatial filter (compute features)
        number_epochs_test_trial = zeros(1,length(vector_session));
        
        % Vectors of training and test
        train.data=[];
        train.task =[];
        test.data=[];
        test.task=[];
        
        for j=1:length(vector_session)
            for f=1:fbands
                for z=1:floor(length(vector_session(j).task_EEG)/shift)
                    if ismember(n_CSP, [1, 2]) % Alpha band (filters 1 to 4)
                        if z*shift<processing_windows % First epoch (processing window may be larger than shift)
                            data = vector_session(j).data_preprocessed_EEG(n_ch+(f-1)*n_ch+1:n_ch+f*n_ch,1:shift);
                        else
                            data = vector_session(j).data_preprocessed_EEG(n_ch+(f-1)*n_ch+1:n_ch+f*n_ch,z*shift-processing_windows+1:z*shift);
                        end
                    else % Gamma band (filters 5 to 10)
                        if z*shift<processing_windows % First epoch (processing window may be larger than shift)
                            data = vector_session(j).data_preprocessed_EEG(n_ch+(f-1+4)*n_ch+1:n_ch+(f+4)*n_ch,1:shift);
                        else
                            data = vector_session(j).data_preprocessed_EEG(n_ch+(f-1+4)*n_ch+1:n_ch+(f+4)*n_ch,z*shift-processing_windows+1:z*shift);
                        end
                    end
                    data_channels_selected=data(channels,:);
                    Z=W_CSP{k,f}*data_channels_selected;
                    % Only consider the variance of the m signals most suitable for discrimination (first m and last m rows of Z).
                    var_Zp= zeros(1,2*m);
                    Zp = [Z(1:m,:); Z(end-m+1:end,:)];
                    for p=1:2*m
                        var_Zp(p) = var(Zp(p,:));
                    end

                    % Normalize variance of each signal
                    features((f-1)*2*m+1:f*2*m,z)=log10(var_Zp'/sum(var_Zp));
                end
            end
            
            % Divide in training and test
            if ~(j == k) % Training
                data_balanced =features(:,epochs_training{j})';
                task_balanced = tasks{j}(epochs_training{j});
                train.data = [train.data;data_balanced];
                train.task = [train.task,task_balanced];
            else % Test
                test.data = [test.data;features(:,epochs_valid{j})'];
                test.task = [test.task tasks{j}(epochs_valid{j})];
            end
        end
        % END Apply spatial filter (compute features)
        %% Classification
        CSP_classify{n_CSP, k} = classificator_lda(test.data,train.data,train.task)';
    end 
end

%% Cascode classification
MI_classify = cell(1, length(vector_session));
att_classify = cell(1, length(vector_session));

for j=1:length(vector_session)
    MI_classify{j} = NaN(1, floor(length(vector_session(j).task_EEG)/shift));
    MI_classify{j}(CSP_classify{1, j} == 0) = 0;
    MI_classify{j}(CSP_classify{1, j} == 1) = CSP_classify{2, j}(CSP_classify{1, j} == 1);
    
    att_classify{j} = NaN(1, floor(length(vector_session(j).task_EEG)/shift));
    att_classify{j}(CSP_classify{3, j} == 0) = 0;
    att_classify{j}(CSP_classify{3, j} == 1) = CSP_classify{4, j}(CSP_classify{3, j} == 1);
end

%% Index computation
for j=1:length(vector_session)
    MI_index{j} = NaN(1, length(MI_classify{j}));
    MI_clas_aux = [zeros(1, 5) MI_classify{j} zeros(1, 4)];
    
    att_index{j} = NaN(1, length(att_classify{j}));
    att_clas_aux = [zeros(1, 5) att_classify{j} zeros(1, 4)];
    
    for z=1:length(MI_index{j})
        MI_index{j}(z) = mean(MI_clas_aux(z:z+9));
        att_index{j}(z) = mean(att_clas_aux(z:z+9));
    end
end

%% Plot results
if plot_results == 1
    figure_att = figure;
    figure_MI = figure;
    sampling_shift_ratio = fm/shift;
    for j=1:length(vector_session)
        idx_count = find(ismember(tasks_epoch{j}, [406, 756, 757, 758]));
        idx_MI = find(ismember(tasks_epoch{j}, [404, 752, 753, 754]));
        idx_relax = find(ismember(tasks_epoch{j}, [402]));
        idx_prepare = find(ismember(tasks_epoch{j}, [405, 403, 401, 400, 751, 755, 761, 759, 760]));
        
        % Attention plot
        figure(figure_att);
        subplot(4,4,j);
        plot(0:1/sampling_shift_ratio:length(tasks_epoch{j})/sampling_shift_ratio-1/sampling_shift_ratio, ismember(tasks_epoch{j}, [403, 404, 751, 752, 753, 754, 761]), 'k-','linewidth',5)
        hold on

        title(sprintf('Trial %d', j))
        plot(0:1/sampling_shift_ratio:length(tasks_epoch{j})/sampling_shift_ratio-1/sampling_shift_ratio, att_index{j}(1,:), 'c-')

        if ~isempty(idx_count)
            plot(idx_count/sampling_shift_ratio- 1/sampling_shift_ratio, 0,'b.','MarkerSize',15)
        end
        if ~isempty(idx_MI)
            plot(idx_MI/sampling_shift_ratio- 1/sampling_shift_ratio, 1, 'r.','MarkerSize',15)
        end
        plot(idx_relax/sampling_shift_ratio - 1/sampling_shift_ratio, 0,'g.','MarkerSize',15)
        plot(idx_prepare/sampling_shift_ratio- 1/sampling_shift_ratio, ismember(tasks_epoch{j}(idx_prepare), [403, 404, 751, 752, 753, 754, 761]),'y.','MarkerSize',15)
        
        xlim([0 length(tasks_epoch{j})/sampling_shift_ratio]);
        ylim([-0.1 1.1]);
        
        % MI plot
        figure(figure_MI);
        subplot(4,4,j);
        plot(0:1/sampling_shift_ratio:length(tasks_epoch{j})/sampling_shift_ratio-1/sampling_shift_ratio,ismember(tasks_epoch{j}, [403, 404, 751, 752, 753, 754, 761]), 'k-','linewidth',5)
        hold on

        title(sprintf('Trial %d', j))
        plot(0:1/sampling_shift_ratio:length(tasks_epoch{j})/sampling_shift_ratio-1/sampling_shift_ratio, MI_index{j}(1,:), 'm-')
        
        if ~isempty(idx_count)
            plot(idx_count/sampling_shift_ratio - 1/sampling_shift_ratio, 0,'b.','MarkerSize',15)
        end
        if ~isempty(idx_MI)
            plot(idx_MI/sampling_shift_ratio- 1/sampling_shift_ratio, 1, 'r.','MarkerSize',15)
        end
        plot(idx_relax/sampling_shift_ratio- 1/sampling_shift_ratio, 0,'g.','MarkerSize',15)
        plot(idx_prepare/sampling_shift_ratio- 1/sampling_shift_ratio, ismember(tasks_epoch{j}(idx_prepare), [403, 404, 751, 752, 753, 754, 761]),'y.','MarkerSize',15)
        
        xlim([0 length(tasks_epoch{j})/sampling_shift_ratio]);
        ylim([-0.1 1.1]);
    end
    figure(figure_att);
    sgtitle('Attention index');
    figure(figure_MI);
    sgtitle('Motor imagery index');
end

%% Compute metrics
results = NaN(14, length(vector_session));
for j=1:length(vector_session)
    % MI area
    results(2, j) = mean(1 - MI_index{j}(ismember(tasks_epoch{j}, relax_class)), 'omitnan'); % Relax
    results(3, j) = mean(MI_index{j}(ismember(tasks_epoch{j}, MI_class)), 'omitnan'); % MI
    results(4, j) = mean(1 - MI_index{j}(ismember(tasks_epoch{j}, count_class)), 'omitnan'); % Count
    results(1, j) = mean([results(2, j), results(3, j), results(4, j)], 'omitnan');
    % Att area
    results(6, j) = mean(1 - att_index{j}(ismember(tasks_epoch{j}, relax_class)), 'omitnan'); % Relax
    results(7, j) = mean(att_index{j}(ismember(tasks_epoch{j}, MI_class)), 'omitnan'); % MI
    results(8, j) = mean(1 - att_index{j}(ismember(tasks_epoch{j}, count_class)), 'omitnan'); % Count
    results(5, j) = mean([results(6, j), results(7, j), results(8, j)], 'omitnan'); % Relax
    
    % MI accuracy
    results(9, j) = mean(ismember(tasks_epoch{j}, [404, 752, 753, 754]) == MI_classify{j}, 'omitnan'); % Total accuracy
    results(10, j) = mean(MI_classify{j}(ismember(tasks_epoch{j}, [402, 406, 756, 757, 758])) == 0, 'omitnan'); % Accuracy during not MI task
    results(11, j) = mean(MI_classify{j}(ismember(tasks_epoch{j}, [404, 752, 753, 754])) == 1, 'omitnan'); % Accuracy during MI task 
    
    % Att accuracy
    results(12, j) = mean(ismember(tasks_epoch{j}, [404, 752, 753, 754]) == att_classify{j}, 'omitnan'); % Total accuracy
    results(13, j) = mean(att_classify{j}(ismember(tasks_epoch{j}, [402, 406, 756, 757, 758])) == 0, 'omitnan'); % Accuracy during not MI task
    results(14, j) = mean(att_classify{j}(ismember(tasks_epoch{j}, [404, 752, 753, 754])) == 1, 'omitnan'); % Accuracy during MI task 
    
end

%% Save results
cd(save_dir);
save_name = input('How do you want the folder of results to be called?: ', 's');
mkdir(save_name);
cd(save_name);
if plot_results == 1
    savefig(figure_att, 'attention_index.fig');
    savefig(figure_MI, 'MI_index.fig');
end
save('metrics.mat', 'results');
save('attention_index.mat', 'att_index');
save('MI_index.mat', 'MI_index');
save_results(att_index, MI_index, tasks_epoch);
save('tasks.mat', 'tasks_epoch');


%% End script
cd(initial_path)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Function definition %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ses = load_session(p)
    % Load data from csv file
     t = readtable(p);
     ses.time = t.time';
     ses.task_EEG = t.task';
     ses.electrodes = t(:, 2:32).Properties.VariableNames;
     ses.data_EEG = table2array(t(:, 2:32))';
end

function numbers=match_electrodes(general_names,specific_names)
    % This function returns the indexes of the specific_names electrodes respect to the general_names
    numbers=zeros(1,length(specific_names));
    for a=1:length(specific_names)
        for b=1:length(general_names)
            if strcmp(specific_names{a},general_names{b})
                numbers(a)=b;
                break;
            end
        end
    end
    return
end

function output = band_pass_filter_setup(fm, number_channels,filt_order,low_f,high_f)
    %Filter frequencies outisde band selected
    %Input: original fm, number channels, Butterworth filter order, low andhigh cut frequencies
    %Output are the parameters of the filter 

    fnq= fm/2;
    Wn    = [low_f high_f];  % Corner cut-off frequencies [Hz]
    Wn1 = Wn/fnq;
    [A,B,C,D] = butter(filt_order,Wn1,'bandpass');

    Xnn=cell(1,number_channels); %state vector

    for k=1:number_channels
        Xnn{k}=zeros(size(A,1),1);
    end
    output = {};
    output.A = A;
    output.B = B;
    output.C = C;
    output.D = D;
    output.Xnn = Xnn;
end

function [data_filtered, Xnn] = band_pass_filter_multiple(data, BP_conf)
    % Apply band pass filter and updates the parameters in BP_conf
    % data: [number of channels x shift samples]
    % data_filter: [number of channels x shift samples]
    % Xnn: update state parameters cell [1 x number of channels])
    % A,B,C,D parameters of the state-space filter

    
    %Band pass filter parameters
    A = BP_conf.filter_parameters.A;
    B = BP_conf.filter_parameters.B;
    C = BP_conf.filter_parameters.C;
    D = BP_conf.filter_parameters.D;
    Xnn = BP_conf.filter_parameters.Xnn;
    data_filtered = zeros(size(data,1),size(data,2));

    %Filter
    for i=1:size(data,2) %for each sample
        for k=1:size(data,1) %for each channel
            Xn1=A*Xnn{k} + B*data(k,i);
            data_filtered(k,i)=(C*Xnn{k}+D*data(k,i));
            Xnn{k}=Xn1;
        end 
    end
end

function [W] = csp(X1, X2)
    %% First way
     % function CSP() from new_european
     % same formula as in the paper 'Optimal spatial filtering of single trialEEG during imagined hand movement' but for covariance matrix computation, +1e-20

        % Covariance matrix from each class
        N=length(X1); %number of trials class 1
        for i=1:N
            C1a(:,:,i)= (X1{i}*X1{i}')/trace(X1{i}*X1{i}'+1e-20);
        end
        N=length(X2); %number of trials class 2
        for i=1:N
            C2a(:,:,i)= (X2{i}*X2{i}')/trace(X2{i}*X2{i}'+1e-20);
        end

        C1=mean(C1a,3); % mean of covariance matrixes for class 1, C1~[C x C]
        C2=mean(C2a,3);% mean of covariance matrixes for class 1, C1~[C x C]


        %Composite spatial covariance
        Cc=C1+C2;


        [V D]=eigs(Cc, rank(Cc));
        P=sqrtm(inv(D))*V';
        S1=P*C1*P';
        S2=P*C2*P';
        [V1 D1]=eigs(S1, size(S1,2));
        [V2 D2]=eigs(S2, size(S2,2),'SM');    % it must be D1+D2=I and V1=V2
        W=V2'*P;
 
end

function prediccion = classificator_lda(test_data,train_data,train_tasks)
    % Linear discriminant classifier
    % test_data and train_data must have the same number of columns (features)
    % train_task is a vector that must have the same number of elements as the rows of train_data
    prediccion=zeros(size(test_data,1),1);
    for i=1:size(test_data,1)
        sample=test_data(i,:);
        prediccion(i)=classify(sample, train_data,train_tasks, 'linear', 'empirical');
    end
end

function save_results(att_index, MI_index, tasks)
    % Save attention index
    att_file = table();
    max_epochs = max(cellfun('length', att_index));
    att_file.time = [0:1/2:max_epochs/2-1/2]';
    for n=1:length(att_index)
        NaN_required = max_epochs - length(att_index{n});
        eval(sprintf("att_file.trial_%d = [att_index{n}'; NaN(NaN_required, 1)];", n));
        eval(sprintf("att_file.trial_%d_task = [tasks{n}'; NaN(NaN_required, 1)];", n));
    end
    writetable(att_file, strcat("attention_index.csv"), "Delimiter", ";");
    
    % Save MI index
    MI_file = table();
    max_epochs = max(cellfun('length', MI_index));
    MI_file.time = [0:1/2:max_epochs/2-1/2]';
    for n=1:length(MI_index)
        NaN_required = max_epochs - length(MI_index{n});
        eval(sprintf("MI_file.trial_%d = [MI_index{n}'; NaN(NaN_required, 1)];", n));
        eval(sprintf("MI_file.trial_%d_task = [tasks{n}'; NaN(NaN_required, 1)];", n));
    end
    writetable(MI_file, strcat("MI_index.csv"), "Delimiter", ";");
end
