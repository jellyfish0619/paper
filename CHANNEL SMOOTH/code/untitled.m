clc
clear all
NumTxAnts = 4; % Number of transmit antennas
NumSTS = 2;    % Number of space-time streams
NumRxAnts = 2; % Number of receive antennas
chanEstSound=zeros(242,NumTxAnts,NumRxAnts);
cfgHEBase = wlanHESUConfig;
cfgHEBase.ChannelBandwidth = 'CBW20';      % Channel bandwidth
cfgHEBase.NumSpaceTimeStreams = NumSTS;    % Number of space-time streams
cfgHEBase.NumTransmitAntennas = NumTxAnts; % Number of transmit antennas
cfgHEBase.APEPLength = 1e3;                % Payload length in bytes
cfgHEBase.ExtendedRange = false;           % Do not use extended range format
cfgHEBase.Upper106ToneRU = false;          % Do not use upper 106 tone RU
cfgHEBase.PreHESpatialMapping = false;     % Spatial mapping of pre-HE fields
cfgHEBase.GuardInterval = 3.2;             % Guard interval duration
cfgHEBase.HELTFType = 4;                   % HE-LTF compression mode
cfgHEBase.ChannelCoding = 'BCC';          % Channel coding
cfgHEBase.MCS = 7;                         % Modulation and coding scheme
cfgHEBase.SpatialMapping = 'Custom';       % Custom for beamforming

%% Null Data Packet (NDP) Configuration
% Configure the NDP transmission to have data length of zero. Since the NDP
% is used to obtain the channel state information, set the number of
% space-time streams equal to the number of transmit antennas and directly
% map each space-time stream to a transmit antenna.



%% Channel Configuration
% This example uses a TGax NLOS indoor channel model with delay profile
% Model-B. The Model-B profile is considered NLOS when the distance between
% transmitter and receiver is greater than or equal to five meters. For
% more information, see
% <docid:wlan_ref#mw_43b5900e-69e1-4636-b084-1e72dbd46293 wlanTGaxChannel>.

% Create and configure the TGax channel
chanBW = cfgHEBase.ChannelBandwidth;
tgaxChannel = wlanTGaxChannel;
tgaxChannel.DelayProfile = 'Model-D';
tgaxChannel.NumTransmitAntennas = NumTxAnts;
tgaxChannel.NumReceiveAntennas = NumRxAnts;
tgaxChannel.TransmitReceiveDistance = 1; % Distance in meters for NLOS
tgaxChannel.ChannelBandwidth = chanBW;
tgaxChannel.LargeScaleFadingEffect = 'None';
tgaxChannel.NormalizeChannelOutputs = false;
tgaxChannel.RandomStream="mt19937ar with seed";
tgaxChannel.Seed=70;
fs = wlanSampleRate(cfgHEBase);
tgaxChannel.SampleRate = fs;
ofdmInfo = wlanHEOFDMInfo('HE-Data',cfgHEBase);

snr = 20:2:28;

maxNumErrors = 100;   % The maximum number of packet errors at an SNR point
maxNumPackets = 1000; % The maximum number of packets at an SNR point


s = numel(snr); % Number of SNR points
packetErrorRate_sumwhole=zeros(s,1);
packetErrorRate_sumtest=zeros(s,1);
packetErrorRate_sumcon=zeros(s,1);

for i=1:s
% Get occupied subcarrier indices and OFDM parameters
packetSNR = snr(i)-10*log10(ofdmInfo.FFTLength/ofdmInfo.NumTones); %控制packetSNR，由i来控制
stream = RandStream('combRecursive','Seed',100);
        stream.Substream = s;
        RandStream.setGlobalStream(stream);
ofdmInfo = wlanHEOFDMInfo('HE-Data',cfgHEBase);
cfgNDP = cfgHEBase;
cfgNDP.APEPLength = 0;                  % NDP has no data
cfgNDP.NumSpaceTimeStreams = NumTxAnts; % For feedback matrix calculation
cfgNDP.SpatialMapping = 'Direct';       % Each TxAnt carries a STS
% Indices to extract fields from the PPDU
ind = wlanFieldIndices(cfgHEBase);
indSound = wlanFieldIndices(cfgNDP);



 tx = wlanWaveformGenerator([],cfgNDP);
            
            % Add trailing zeros to allow for channel delay
txPad = [tx; zeros(50,cfgNDP.NumTransmitAntennas)];
            
            % Pass through a fading indoor TGax channel
 reset(tgaxChannel); % Reset channel for different realization
 rx = tgaxChannel(txPad);

            % Pass the waveform through AWGN channel
 rx = awgn(rx,packetSNR);
            
            % Calculate the steering matrix at the beamformee
 V = heUserBeamformingFeedback(rx,cfgNDP,true);
 V = permute(V,[3 2 1]);
 V_test=V;
         [V_test,a_test, ac_test,Q_test]=smooth_test(V_test,chanEstSound);

         V_whole=V;
         [V_whole,ac_whole,a_whole,Q_whole,infobb]=smooth_manifoldswhole(V_test,chanEstSound,Q_test);
           infowhole(i,1:length([infobb.cost]))=[infobb.cost];
T_whole=permute(V_whole(:,1:NumSTS,:),[3 2 1]);
T_test=permute(V_test(:,1:NumSTS,:),[3,2,1]);
T=permute(V(:,1:NumSTS,:),[3,2,1]);

%whole
 cfgHE_wh=cfgHEBase;
 cfgHE_wh.SpatialMappingMatrix = T_whole;
 psduLength = getPSDULength(cfgHE_wh); % PSDU length in bytes
 txPSDU = randi([0 1],psduLength*8,1); % Generate random PSDU
%test
 cfgHE_test=cfgHEBase;
 cfgHE_test.SpatialMappingMatrix=T_test;

%conventional
cfgHE_con=cfgHEBase;
cfgHE_con.SpatialMappingMatrix=T;

        [numPacketErrors_whole,numPkt_whole]=PERTest_HE(cfgHE_wh,txPSDU,tgaxChannel,maxNumErrors,maxNumPackets,packetSNR,ind,fs);
        [numPacketErrors_test,numPkt_test]=PERTest_HE(cfgHE_test,txPSDU,tgaxChannel,maxNumErrors,maxNumPackets,packetSNR,ind,fs);
        [numPacketErrors_con,numPkt_con]=PERTest_HE(cfgHE_con,txPSDU,tgaxChannel,maxNumErrors,maxNumPackets,packetSNR,ind,fs); 
        packetErrorRate_whole(i)=numPacketErrors_whole/(numPkt_whole-1);
        packetErrorRate_sumwhole(i)=packetErrorRate_sumwhole(i)+packetErrorRate_whole(i);
        packetErrorRate_test(i)=numPacketErrors_test/(numPkt_test-1);
        packetErrorRate_sumtest(i)=packetErrorRate_sumtest(i)+packetErrorRate_test(i);
        packetErrorRate_con(i)=numPacketErrors_con/(numPkt_con-1);
        packetErrorRate_sumcon(i)=packetErrorRate_sumcon(i)+packetErrorRate_con(i);
        disp(['smooth-BF-whole:SNR ' num2str(snr(i)) ' completed after ' ...
            num2str(numPkt_whole-1) ' packets, PER: ' ...
            num2str(packetErrorRate_whole(i))]);%
        disp(['test:SNR ' num2str(snr(i)) ' completed after ' ...
            num2str(numPkt_test-1) ' packets, PER: ' ...
            num2str(packetErrorRate_test(i))]);%
        disp(['convention:SNR ' num2str(snr(i)) ' completed after ' ...
            num2str(numPkt_con-1) ' packets, PER: ' ...
            num2str(packetErrorRate_con(i))]);%
subcarrierNum=242;
figure;
plot((1:subcarrierNum-1)',abs(a_test));
hold on;
plot((1:subcarrierNum-1)',abs(ac_test));
hold on;
plot((1:subcarrierNum-1)',abs(ac_whole));
legend('优化前','原文','整体流行优化');
hold off;

end




