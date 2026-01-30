function [numPacketErrors,numPkt]=PERTest_HE(cfgHE,txPSDU,tgaxChannel,maxNumErrors,maxNumPackets,packetSNR,ind,fs)                
numPacketErrors=0;
numPkt=1;
seed=1;
a=zeros(242,1);
chanBW=cfgHE.ChannelBandwidth;
ofdmInfo = wlanHEOFDMInfo('HE-Data',cfgHE);

 while (numPacketErrors<=maxNumErrors && numPkt<=maxNumPackets)
            seed=seed+1;
             tx = wlanWaveformGenerator(txPSDU,cfgHE);
            
            % Add trailing zeros to allow for channel delay
            txPad = [tx; zeros(50,cfgHE.NumTransmitAntennas)];
            reset(tgaxChannel);
            % Pass through a fading indoor TGax channel
            rx = tgaxChannel(txPad);
            rng(seed);
            % Pass the waveform through AWGN channel
            rx = awgn(rx,packetSNR);
             

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
            rxPSDU = wlanHEDataBitRecover(eqDataSym,nVarEst,csi,cfgHE,'LDPCDecodingMethod','norm-min-sum');
            
            % Determine if any bits are in error, i.e. a packet error
            packetError = ~isequal(txPSDU,rxPSDU);
            numPacketErrors = numPacketErrors+packetError;
            numPkt = numPkt+1;
                
 end