
clear all
NumTxAnts = 4; % Number of transmit antennas
NumSTS = 4;    % Number of space-time streams
NumRxAnts = 4; % Number of receive antennas
seed=123;
% % Create a format configuration object for a 8-by-8 VHT transmission
% cfgVHT = wlanVHTConfig;
% cfgVHT.ChannelBandwidth = 'CBW80'; % 80 MHz channel bandwidth
% cfgVHT.NumTransmitAntennas = NumTxAnts;    % 4 transmit antennas
% cfgVHT.NumSpaceTimeStreams = NumStreams;    % 1 space-time streams  
% cfgVHT.APEPLength = 2000;          % APEP length in bytes
% cfgVHT.MCS = 7;                    % 256-QAM rate-5/6
% cfgVHT.ChannelCoding = 'BCC';      % Binary convolutional coding
% % Create and configure the channel

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
chanBW = cfgHEBase.ChannelBandwidth;
tgaxChannel = wlanTGaxChannel;
tgaxChannel.DelayProfile = 'Model-D';
tgaxChannel.NumReceiveAntennas = 4;
tgaxChannel.TransmitReceiveDistance = 1; % Distance in meters for NLOS
tgaxChannel.ChannelBandwidth = cfgHEBase.ChannelBandwidth;
tgaxChannel.NumTransmitAntennas = 4;
tgaxChannel.RandomStream="mt19937ar with seed";
tgaxChannel.Seed=70;
tgaxChannel.LargeScaleFadingEffect = 'None';
tgaxChannel.NormalizeChannelOutputs = false;
fs = wlanSampleRate(cfgHEBase);
% Get the OFDM info
ofdmInfo = wlanHEOFDMInfo('HE-Data',cfgHEBase);
% Set the sampling rate of the channel
snr = 29:1:35;
maxNumErrors = 10;   % The maximum number of packet errors at an SNR point
maxNumPackets =100; % The maximum number of packets at an SNR point
packetError=zeros(maxNumPackets);
% S = RandStream('mt19937ar',Seed=0);
cfgNDP = cfgHEBase;
cfgNDP.APEPLength = 0;                  % NDP has no data
cfgNDP.NumSpaceTimeStreams = NumTxAnts; % For feedback matrix calculation
cfgNDP.SpatialMapping = 'Direct';       % Each TxAnt carries a
%不一样
% release(tgacChannel);
tgaxChannel.SampleRate = fs;

% Indices for accessing each field within the time-domain packet
% ind = wlanFieldIndices(cfgVHT);
% s = numel(snr);
ind = wlanFieldIndices(cfgHEBase);
indSound = wlanFieldIndices(cfgNDP);
s = numel(snr);

%parfor i = 1:S % Use 'parfor' to speed up the simulation
% stream = RandStream('combRecursive';
        cfgNDP = cfgHEBase;
cfgNDP.APEPLength = 0;                  % NDP has no data
cfgNDP.NumSpaceTimeStreams = NumTxAnts; % For feedback matrix calculation
cfgNDP.SpatialMapping = 'Direct';       % Each TxAnt carries a

% 误码率测试
%
for p=1:1  %控制测试的次数（轮数）
    fprintf("第%d轮\n",p);
%     探测信道

    for i = 1:s  %控制SNR的大小的循环

%不一样
        packetSNR = snr(i)-10*log10(ofdmInfo.FFTLength/ofdmInfo.NumTones); %控制packetSNR，由i来控制
        stream = RandStream("combRecursive",'seed',70);
        stream.Substream = i;
        RandStream.setGlobalStream(stream);
% %         探测信道的设置
%         vhtSound = cfgVHT;
%         vhtSound.APEPLength = 0; % NDP so no data
%         vhtSound.SpatialMapping = 'Direct'; % Each TxAnt carries a STS
        cfgHE = cfgHEBase;
%         Generate sounding waveform
        
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

%             if isempty(V)
%                 % User feedback failed, packet error
%                 numPacketErrors = numPacketErrors+1;
%                 numPkt = numPkt+1;
%                 continue; % Go to next loop iteration
%             end
              V= permute(V,[3 2 1]); % Permute to Nr-by-Nt-by-Nst
               %[V,rho_all,rho_allc,Q]=smooth_test(V,4);
                 %V=smooth_manifoldswhole(V,V,4,Q);

              V= permute(V,[3 2 1]); % Permute to Nr-by-Nt-by-Nst
            
            steeringMat = V(:,1:NumSTS,:);



            % Beamformed data transmission
            psduLength = getPSDULength(cfgHE); % PSDU length in bytes
            txPSDU = randi([0 1],psduLength*8,1); % Generate random PSDU
            cfgHE.NumSpaceTimeStreams = NumTxAnts; % For feedback matrix calculation
            cfgHE.SpatialMapping = 'Custom';       % Each TxAnt carries a
           cfgHE.SpatialMappingMatrix = steeringMat;


        numPacketErrors = 0;
        numPkt = 1; % Index of packet transmitted
        while numPacketErrors<=maxNumErrors && numPkt<=maxNumPackets
            % Null data packet transmission
            
            tx = wlanWaveformGenerator(txPSDU,cfgHE);
            
            % Add trailing zeros to allow for channel delay
            txPad = [tx; zeros(50,cfgHE.NumTransmitAntennas)];

            % Pass through a fading indoor TGax channel
            reset(tgaxChannel);
            rx = tgaxChannel(txPad);
            % Pass the waveform through AWGN channel
            
            rx = awgn(rx,packetSNR);
            
            % Packet detect and determine coarse packet offset
            coarsePktOffset = wlanPacketDetect(rx,chanBW);
            if isempty(coarsePktOffset) % If empty no L-STF detected; packet error
                numPacketErrors = numPacketErrors+1;
                numPkt = numPkt+1;
                continue; % Go to next loop iteration
            end

            % Extract L-STF and perform coarse frequency offset correction
            lstf = rx(coarsePktOffset+(ind.LSTF(1):ind.LSTF(2)),:);
            coarseFreqOff = wlanCoarseCFOEstimate(lstf,chanBW);
            rx = frequencyOffset(rx,fs,-coarseFreqOff);

            % Extract the non-HT fields and determine fine packet offset
            nonhtfields = rx(coarsePktOffset+(ind.LSTF(1):ind.LSIG(2)),:);
            finePktOffset = wlanSymbolTimingEstimate(nonhtfields,chanBW);

            % Determine final packet offset
            pktOffset = coarsePktOffset+finePktOffset;

            % If packet detected outwith the range of expected delays from
            % the channel modeling; packet error
            if pktOffset>50
                numPacketErrors = numPacketErrors+1;
                numPkt = numPkt+1;
                continue; % Go to next loop iteration
            end

            % Extract L-LTF and perform fine frequency offset correction
            rxLLTF = rx(pktOffset+(ind.LLTF(1):ind.LLTF(2)),:);
            fineFreqOff = wlanFineCFOEstimate(rxLLTF,chanBW);
            rx = frequencyOffset(rx,fs,-fineFreqOff);

            % HE-LTF demodulation and channel estimation
            rxHELTF = rx(pktOffset+(ind.HELTF(1):ind.HELTF(2)),:);
            heltfDemod = wlanHEDemodulate(rxHELTF,'HE-LTF',cfgHE);
            [chanEst,pilotEst] = wlanHELTFChannelEstimate(heltfDemod,cfgHE);
            
            chanEstPerm = permute(chanEst,[3 2 1]); % Permute to Nr-by-Nt-by-Nst
            [U,D,V] = pagesvd(chanEstPerm,'econ'); %SVD分

            % Data demodulate
            rxData = rx(pktOffset+(ind.HEData(1):ind.HEData(2)),:);
            demodSym = wlanHEDemodulate(rxData,'HE-Data',cfgHE);

                  % Pilot phase tracking
            % Average single-stream pilot estimates over symbols (2nd dimension)
            pilotEstTrack = mean(pilotEst,2);
            demodSym = wlanHETrackPilotError(demodSym,pilotEstTrack,cfgHE,'HE-Data');

            % Estimate noise power in HE fields
            nVarEst = heNoiseEstimate(demodSym(ofdmInfo.PilotIndices,:,:),pilotEstTrack,cfgHE);
            
            % Extract data subcarriers from demodulated symbols and channel
            % estimate
            demodDataSym = demodSym(ofdmInfo.DataIndices,:,:);
            chanEstData = chanEst(ofdmInfo.DataIndices,:,:);
            
            % Equalization and STBC combining
            [eqDataSym,csi] = heEqualizeCombine(demodDataSym,chanEstData,nVarEst,cfgHE);
            
            % Recover data
            rxPSDU = wlanHEDataBitRecover(eqDataSym,nVarEst,csi,cfgHE,'LDPCDecodingMethod','layered-bp');
            
            % Determine if any bits are in error, i.e. a packet error
            packetError = ~isequal(txPSDU,rxPSDU);
            numPacketErrors = numPacketErrors+packetError;
            numPkt = numPkt+1;
        end
        % Calculate packet error rate (PER) at SNR point
        packetErrorRate(i) = numPacketErrors/(numPkt-1);
        disp(['MCS ' num2str(cfgHE.MCS) ','...
              ' SNR ' num2str(snr(i)) ...
              ' completed after ' num2str(numPkt-1) ' packets,'...
              ' PER:' num2str(packetErrorRate(i))]);
    end
    disp(newline);


   

end













