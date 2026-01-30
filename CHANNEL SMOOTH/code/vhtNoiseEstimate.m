function nVarVHT = vhtNoiseEstimate(vhtdata, chanEstSSPilots, vhtBF)
% vhtNoiseEstimate  Estimate noise variance using pilot channel estimate time-differences
%   nVarVHT = vhtNoiseEstimate(vhtdata, chanEstSSPilots, vhtBF)
%
% Inputs (only chanEstSSPilots is required by this implementation):
%   vhtdata          : Nst x Nsym [x Nr]  (unused here; kept for API compatibility)
%   chanEstSSPilots  : Np  x Nsym x Nr    (pilot-only single-stream channel estimates)
%   vhtBF            : wlanVHTConfig or any; (unused here; kept for API compatibility)
%
% Output:
%   nVarVHT          : scalar noise variance estimate (linear power)
%
% Method:
%   Assuming channel is quasi-static across adjacent OFDM symbols, the time
%   difference of pilot channel estimates is dominated by noise:
%       err = H(:,t,:) - H(:,t-1,:)
%   Var(err) ~= 2*sigma^2  -->  sigma^2 ~= mean(|err|^2)/2
%
% Notes:
%   - This method does not rely on pilot indices or pilot sign sequences.
%   - It is robust when Nsym >= 2; with Nsym == 1 we fall back to a
%     spatial de-mean across pilots (weaker assumption).

    %#ok<*INUSD> % keep inputs for signature compatibility

    H = chanEstSSPilots;       % Np x Nsym x Nr (complex double)
    sz = size(H);
    if numel(sz) < 3
        % Ensure a 3-D array: Np x Nsym x Nr
        H = reshape(H, sz(1), sz(2), 1);
        sz = size(H);
    end

    Np   = sz(1);
    Nsym = sz(2);
    Nr   = sz(3);

    if Nsym >= 2
        % Time-difference across adjacent OFDM symbols
        % err size: Np x (Nsym-1) x Nr
        err = diff(H, 1, 2);
        % Noise variance estimate (scalar)
        nVarVHT = mean(abs(err).^2, 'all') / 2;
    else
        % Fallback when only a single symbol is available:
        % De-mean across pilots per antenna, treat residual as noise proxy.
        % This is weaker but avoids extra dependencies.
        H1 = H(:,1,:);                            % Np x 1 x Nr
        H1 = H1 - mean(H1, 1);                    % remove common mode
        nVarVHT = mean(abs(H1).^2, 'all');        % proxy for sigma^2
    end

    % Safety: ensure real, non-negative scalar
    nVarVHT = max(real(nVarVHT), 0);
end
