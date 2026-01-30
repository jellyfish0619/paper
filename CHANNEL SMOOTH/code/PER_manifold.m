
function [packetErrorRate_sumsmooth,a,as]=PER_manifold() 
clear all
NumTxAnts=4;
NumSTS=4;
NumStreams=4;
seed=123;
% Create a format configuration object for a 8-by-8 VHT transmission
cfgVHT = wlanVHTConfig;
cfgVHT.ChannelBandwidth = 'CBW80'; % 80 MHz channel bandwidth
cfgVHT.NumTransmitAntennas = NumTxAnts;    % 4 transmit antennas
cfgVHT.NumSpaceTimeStreams = NumStreams;    % 1 space-time streams
cfgVHT.APEPLength = 2000;          % APEP length in bytes
cfgVHT.MCS = 7;                    % 256-QAM rate-5/6
cfgVHT.ChannelCoding = 'BCC';      % Binary convolutional coding
% Create and configure the channel
tgacChannel = wlanTGaxChannel;
tgacChannel.DelayProfile = 'Model-D';
tgacChannel.NumReceiveAntennas = NumSTS;
tgacChannel.TransmitReceiveDistance = 1; % Distance in meters for NLOS
tgacChannel.ChannelBandwidth = cfgVHT.ChannelBandwidth;
tgacChannel.NumTransmitAntennas = cfgVHT.NumTransmitAntennas;
tgacChannel.RandomStream="mt19937ar with seed";
tgacChannel.Seed=70;
tgacChannel.LargeScaleFadingEffect = 'None';
tgacChannel.NormalizeChannelOutputs = false;
fs = wlanSampleRate(cfgVHT);
% Get the OFDM info
ofdmInfo = wlanVHTOFDMInfo('VHT-Data',cfgVHT);
% Set the sampling rate of the channel
snr = 34:2:44;
maxNumErrors = 10;   % The maximum number of packet errors at an SNR point
maxNumPackets =100; % The maximum number of packets at an SNR point
% S = RandStream('mt19937ar',Seed=0);

%不一样
% release(tgacChannel);
tgacChannel.SampleRate = fs;

% Indices for accessing each field within the time-domain packet
ind = wlanFieldIndices(cfgVHT);
s = numel(snr);
packetErrorRate_sumsmooth = zeros(s,1);


%parfor i = 1:S % Use 'parfor' to speed up the simulation
% stream = RandStream('combRecursive';


% 误码率测试
%
for p=1:1  %控制测试的次数（轮数）
    fprintf("第%d轮\n",p);
%     探测信道

    for i = 1:s  %控制SNR的大小的循环
        seed=seed+1;
        packetSNR = snr(i)-10*log10(ofdmInfo.FFTLength/ofdmInfo.NumTones); %控制packetSNR，由i来控制
        stream = RandStream("combRecursive",'seed',79);
        stream.Substream = i;
        RandStream.setGlobalStream(stream);
%         探测信道的设置
        vhtSound = cfgVHT;
        vhtSound.APEPLength = 0; % NDP so no data
        vhtSound.SpatialMapping = 'Direct'; % Each TxAnt carries a STS

%         Generate sounding waveform
        soundingPSDU = [];
        tx_detect = wlanWaveformGenerator(soundingPSDU,vhtSound);
        reset(tgacChannel);
        rx_detect = tgacChannel([tx_detect; zeros(15,NumTxAnts)]);
        %reset(tgacChannel);
        rng(seed);
        rx_detect = awgn(rx_detect,packetSNR);
        tOff = wlanSymbolTimingEstimate(rx_detect(ind.LSTF(1):ind.LSIG(2),:),vhtSound.ChannelBandwidth);
        vhtLLTFInd = wlanFieldIndices(vhtSound,'VHT-LTF');
        vhtltf = rx_detect(tOff+(vhtLLTFInd(1):vhtLLTFInd(2)),:);
        vhtltfDemod_test = wlanVHTLTFDemodulate(vhtltf,vhtSound);
%         到此结束探测信道的设置
%         探测信道估计
        chanEstSound = wlanVHTLTFChannelEstimate(vhtltfDemod_test,vhtSound,5);%得到信道估计 
        chanEstSound = vhtBeamformingRemoveCSD(chanEstSound, ...
            vhtSound.ChannelBandwidth,vhtSound.NumSpaceTimeStreams);%移除影响

%         获取信道估计矩阵的大小
        [m, n, u] = size(chanEstSound);

%         对平滑后的信道估计进行SVD分解
        chanEstPerm = permute(chanEstSound,[3 2 1]); % Permute to Nr-by-Nt-by-Nst
        [U,D,V] = pagesvd(chanEstPerm,'econ'); %SVD分解

%         chanEstPerm_c = permute(filtered_estimation,[3 2 1]); % Permute to Nr-by-Nt-by-Nst
%         [U_c,D_c,V_c] = pagesvd(chanEstPerm_c,'econ'); %SVD分解

        

%         进行流行优化
        tic
        V_smooth=V;
        [V_smooth,as, a, Q_all]=smooth_manifolds(V_smooth,chanEstSound,NumTxAnts);
        toc
        disp(['smooth运行时间：',num2str(toc)]);

   
%     得出转向矩阵T
        T_sm = Turn(V_smooth,NumStreams); % Permute to Nst-by-Nsts-by-Nt


%         平滑后的BF信道的设置
        vhtBF_smooth=cfgVHT;
        vhtBF_smooth.SpatialMapping = 'Custom';
        vhtBF_smooth.SpatialMappingMatrix = T_sm; 


%         误码率的仿真

        txPSDU=zeros(cfgVHT.PSDULength*8,1);
        

        [numPacketErrors_smooth,numPkt_smooth]=PERTest(vhtBF_smooth,txPSDU,tgacChannel,maxNumErrors,maxNumPackets,packetSNR,ind,fs);
          

          
        packetErrorRate_smoooth(i)=numPacketErrors_smooth/(numPkt_smooth-1);
        
       
        packetErrorRate_sumsmooth(i)=packetErrorRate_sumsmooth(i)+packetErrorRate_smoooth(i);
       
        disp(['smooth-BF:SNR ' num2str(snr(i)) ' completed after ' ...
            num2str(numPkt_smooth-1) ' packets, PER: ' ...
            num2str(packetErrorRate_smoooth(i))]);


    end
end


% [subcarrierNum,~,~] = size(chanEstSound);
% figure
% plot((1:subcarrierNum-1)',abs(a));
% hold on;
% plot((1:subcarrierNum-1)',abs(as));
% legend('优化前','优化后');
% title('manifold');
% hold off;




