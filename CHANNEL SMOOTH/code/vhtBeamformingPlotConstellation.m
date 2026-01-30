function constDiag = vhtBeamformingPlotConstellation(sym,refSym,titleStr,varargin)
% VHTBeamformingExample Featured example helper function
% Plot constellation

%   Copyright 2015-2019 The MathWorks, Inc.

if nargin>3
    % Channel names specified as optional argument. Assume sym input is
    % already in correct format for plotting
    channelNames = varargin{1};
    symPlot = sym; % Assume each column is a channel to plot
else
    % Assume sym input is Nsd-by-Nsym-by-Nss
    [Nsd,NSym,Nss] = size(sym);
    symPlot = squeeze(reshape(sym(:,:,end:-1:1),Nsd*NSym,1,Nss));
    channelNames = arrayfun(@(x)['Spatial stream ' num2str(x)],1:Nss,'UniformOutput',false);
end

constDiag = comm.ConstellationDiagram;
constDiag.ShowReferenceConstellation = true;
constDiag.ReferenceConstellation = refSym;
constDiag.ShowLegend = true;
constDiag.ChannelNames = channelNames;
constDiag.Title = titleStr;
constDiag.XLimits = [-2 2];
constDiag.YLimits = [-2 2];
% Use a channel per spatial stream
constDiag(symPlot);
release(constDiag);

end