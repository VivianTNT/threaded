import { ExternalLink } from "lucide-react"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { Button } from "@/components/ui/button"

interface LinksPopoverProps {
  urls: string[]
}

export function LinksPopover({ urls }: LinksPopoverProps) {
  if (!urls || urls.length === 0) {
    return <span className="text-sm text-muted-foreground">N/A</span>
  }

  return (
    <Popover>
      <PopoverTrigger asChild onClick={(e) => e.stopPropagation()}>
        <Button variant="link" className="h-auto p-0 text-blue-600 hover:underline text-sm flex items-center gap-1">
          <ExternalLink className="h-3 w-3" />
          {urls.length === 1 ? 'Link' : `${urls.length} links`}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80" onClick={(e) => e.stopPropagation()}>
        <div className="space-y-2">
          <h4 className="font-medium text-sm">{urls.length === 1 ? 'Link' : 'All Links'}</h4>
          <div className="space-y-1 max-h-60 overflow-y-auto">
            {urls.map((url, index) => (
              <a
                key={index}
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline text-xs flex items-start gap-1 p-1 hover:bg-muted rounded"
                onClick={(e) => e.stopPropagation()}
              >
                <ExternalLink className="h-3 w-3 mt-0.5 flex-shrink-0" />
                <span className="break-all">{url}</span>
              </a>
            ))}
          </div>
        </div>
      </PopoverContent>
    </Popover>
  )
}
